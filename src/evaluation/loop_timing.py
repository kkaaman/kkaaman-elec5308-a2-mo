"""Loop-timing profiles from retained run diagnostics (TEST-22, TEST-17).

Two retained formats are understood, both written on one host monotonic clock:

* an S0 run folder from ``carla_adapter.s0_runner`` (``samples.csv`` with
  ``capture_to_apply_ms``, ``frame_age`` and ``real_time_factor``); and
* a ``timing.jsonl`` from ``scripts/measure_operational_governance.py`` or
  ``scripts/capture_operational_approach.py`` with per-stage seconds
  (``sensor_availability_s``, ``bundle_s``, ``detector_s``, ``filter_s``,
  ``governor_s``, ``controller_s``, ``decision_and_overhead_s``) and the
  end-to-end ``capture_to_decision_s`` / ``receipt_to_decision_s``.

Rows may also be built from ``autonomy.interfaces.Diagnostics`` objects with
:func:`rows_from_diagnostics`. All durations are reported in milliseconds;
percentiles use linear interpolation at index ``(n - 1) * p`` like
``evaluation.s0_metrics``. Values are measurements; the only verdict is the
``capture-to-apply p95 <= budget`` gate, which is INSUFFICIENT_EVIDENCE below
``min_samples`` rows rather than PASS.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

__all__ = [
    "DEFAULT_BUDGET_MS",
    "END_TO_END_KEYS",
    "STAGE_KEYS",
    "compare_profiles",
    "load_timing_rows",
    "profile_rows",
    "rows_from_diagnostics",
]

DEFAULT_BUDGET_MS = 100.0
MIN_SAMPLES = 20
# Known per-stage wall durations in seconds, in pipeline order.
STAGE_KEYS = (
    "sensor_availability_s",
    "bundle_s",
    "detector_s",
    "filter_s",
    "governor_s",
    "controller_s",
    "decision_and_overhead_s",
    "stack_s",
    "segmentation_s",
    "lane_monitor_s",
    "lidar_s",
    "hud_s",
)
# End-to-end definitions in preference order: (key, unit scale to ms, definition).
END_TO_END_KEYS = (
    ("capture_to_apply_ms", 1.0, "RGB callback receipt to acknowledged command application"),
    ("capture_to_decision_s", 1000.0, "host clock before tick request to stack decision"),
    ("end_to_end_ms", 1.0, "caller-declared end-to-end latency"),
)
_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


def _finite(value, *, minimum=None):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and (minimum is None or value >= minimum)
    )


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    left, right = math.floor(index), math.ceil(index)
    return ordered[left] + (ordered[right] - ordered[left]) * (index - left)


def _summary_ms(values_ms):
    if not values_ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None, "mean_ms": None}
    return {
        "n": len(values_ms),
        "p50_ms": _percentile(values_ms, 0.5),
        "p95_ms": _percentile(values_ms, 0.95),
        "max_ms": max(values_ms),
        "mean_ms": statistics.mean(values_ms),
    }


def load_timing_rows(path) -> tuple[list[dict], str]:
    """Load rows from a run folder, ``samples.csv``, ``timing.jsonl`` or JSON list.

    Returns ``(rows, source_kind)`` where ``source_kind`` is ``s0_samples``,
    ``timing_jsonl`` or ``json_rows``. Numeric strings from CSV are converted;
    blanks become ``None``. Raises ``FileNotFoundError`` for an unknown layout.
    """
    path = Path(path)
    if path.is_dir():
        if (path / "samples.csv").is_file():
            path = path / "samples.csv"
        elif (path / "timing.jsonl").is_file():
            path = path / "timing.jsonl"
        else:
            raise FileNotFoundError(f"No samples.csv or timing.jsonl in {path}")
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            rows = [
                {key: _number(value) if key != "phase" and key != "status" else value
                 for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        return rows, "s0_samples"
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows, "timing_jsonl"
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON timing input must be a list of rows or {'rows': [...]}")
        return rows, "json_rows"
    raise FileNotFoundError(f"Unsupported timing input: {path}")


def rows_from_diagnostics(diagnostics, end_to_end_s=None) -> list[dict]:
    """Build rows from ``Diagnostics`` objects (``stage_durations_s`` pairs).

    ``end_to_end_s`` optionally supplies one capture-to-apply duration in
    seconds per diagnostics object; it is stored as ``capture_to_apply_ms``.
    Stage names map to ``<name>_s`` columns.
    """
    rows = []
    ends = list(end_to_end_s) if end_to_end_s is not None else [None] * len(diagnostics)
    if len(ends) != len(diagnostics):
        raise ValueError("end_to_end_s must match the diagnostics count")
    for item, end in zip(diagnostics, ends):
        row = {}
        for name, duration in item.stage_durations_s:
            row[f"{name}_s"] = duration
        row["observation_valid"] = item.observation_valid
        if end is not None:
            row["capture_to_apply_ms"] = end * 1000.0
        rows.append(row)
    return rows


def _stage_columns(rows):
    names = [key for key in STAGE_KEYS if any(key in row for row in rows)]
    extra = sorted(
        {
            key
            for row in rows
            for key in row
            if key.endswith("_s")
            and key not in names
            and not key.endswith("_monotonic_s")
            and key
            not in (
                "sim_time_s",
                "simulation_time_s",
                "capture_time_s",
                "capture_to_decision_s",
                "receipt_to_decision_s",
                "sensor_availability_monotonic_s",
            )
            and not key.startswith(("sim_", "simulation_"))
        }
    )
    return names + extra


def _real_time_factor(rows):
    direct = [row.get("real_time_factor") for row in rows]
    direct = [value for value in direct if _finite(value) and value > 0]
    if direct:
        return {
            "min": min(direct),
            "median": statistics.median(direct),
            "n": len(direct),
            "definition": "per-sample real_time_factor column",
        }
    sim_key = next((k for k in ("simulation_time_s", "sim_time_s") if any(k in r for r in rows)), None)
    wall_key = next(
        (
            k
            for k in ("apply_monotonic_s", "decision_monotonic_s", "capture_started_monotonic_s")
            if any(k in r for r in rows)
        ),
        None,
    )
    if sim_key is None or wall_key is None:
        return {"min": None, "median": None, "n": 0, "definition": "not derivable"}
    pairs = [
        (row[sim_key], row[wall_key])
        for row in rows
        if _finite(row.get(sim_key)) and _finite(row.get(wall_key))
    ]
    if len(pairs) < 2:
        return {"min": None, "median": None, "n": 0, "definition": "not derivable"}
    sim_elapsed = pairs[-1][0] - pairs[0][0]
    wall_elapsed = pairs[-1][1] - pairs[0][1]
    factor = sim_elapsed / wall_elapsed if wall_elapsed > 0 else None
    return {
        "min": factor,
        "median": factor,
        "n": len(pairs),
        "definition": f"({sim_key} span) / ({wall_key} span) over the whole record",
    }


def profile_rows(rows, *, label="run", budget_ms=DEFAULT_BUDGET_MS, min_samples=MIN_SAMPLES) -> dict:
    """Per-stage and end-to-end p50/p95 in ms, real-time factor and the budget gate.

    ``budget_ms`` is the frozen capture-to-apply limit (100 ms). The gate is
    ``PASS`` when p95 <= budget with at least ``min_samples`` finite samples,
    ``FAIL`` when p95 exceeds it (any sample count), otherwise
    ``INSUFFICIENT_EVIDENCE``. Missing stages are omitted, not zero.
    """
    if not _finite(budget_ms, minimum=0) or budget_ms <= 0:
        raise ValueError("budget_ms must be positive")
    if type(min_samples) is not int or min_samples < 1:
        raise ValueError("min_samples must be a positive integer")
    rows = [row for row in rows if isinstance(row, dict)]
    end_key = end_scale = definition = None
    for key, scale, text in END_TO_END_KEYS:
        if any(_finite(row.get(key), minimum=0) for row in rows):
            end_key, end_scale, definition = key, scale, text
            break
    end_values = (
        [row[end_key] * end_scale for row in rows if _finite(row.get(end_key), minimum=0)]
        if end_key
        else []
    )
    end = _summary_ms(end_values)
    issues = []
    if not end_values:
        issues.append("No end-to-end latency samples were recorded")
    elif len(end_values) < min_samples:
        issues.append(f"Only {len(end_values)} end-to-end samples; {min_samples} required")
    if end["p95_ms"] is not None and end["p95_ms"] > budget_ms:
        gate = "FAIL"
    elif issues:
        gate = _INSUFFICIENT
    else:
        gate = "PASS"
    stages = {}
    for key in _stage_columns(rows):
        values = [row[key] * 1000.0 for row in rows if _finite(row.get(key), minimum=0)]
        if values:
            stages[key[:-2]] = _summary_ms(values)
    receipt = [
        row["receipt_to_decision_s"] * 1000.0
        for row in rows
        if _finite(row.get("receipt_to_decision_s"), minimum=0)
    ]
    ages = [row.get("frame_age") for row in rows]
    ages = [int(a) for a in ages if _finite(a, minimum=0) and float(a).is_integer()]
    valid_flags = [row.get("observation_valid") for row in rows if "observation_valid" in row]
    return {
        "label": label,
        "sample_count": len(rows),
        "end_to_end": {
            "definition": definition,
            "source_key": end_key,
            "budget_ms": budget_ms,
            "status": gate,
            "evidence_issues": issues,
            **end,
        },
        "receipt_to_decision": _summary_ms(receipt) if receipt else None,
        "stages": stages,
        "real_time_factor": _real_time_factor(rows),
        "frame_age": {"max": max(ages) if ages else None, "n": len(ages)},
        "observation_valid_fraction": (
            sum(bool(flag) for flag in valid_flags) / len(valid_flags) if valid_flags else None
        ),
        "percentile_rule": "linear interpolation at (n - 1) * p",
    }


def compare_profiles(profiles: dict, *, baseline: str, final: str | None = None) -> dict:
    """Relate labelled profiles to ``baseline`` and judge the ``final`` configuration.

    Returns per-label p95 deltas (ms) for end-to-end and shared stages, plus a
    ``gate`` block: ``decision`` is ``ADMIT`` when the final profile's gate is
    PASS, ``BLOCKED`` otherwise, with the reason spelled out. No threshold is
    relaxed here; a missing final profile is BLOCKED with reason.
    """
    if baseline not in profiles:
        raise KeyError(f"Baseline profile {baseline!r} is not present")
    base = profiles[baseline]
    deltas = {}
    for label, profile in profiles.items():
        if label == baseline:
            continue
        stage_delta = {}
        for name, summary in profile["stages"].items():
            other = base["stages"].get(name)
            if other and summary["p95_ms"] is not None and other["p95_ms"] is not None:
                stage_delta[name] = summary["p95_ms"] - other["p95_ms"]
        end_delta = (
            profile["end_to_end"]["p95_ms"] - base["end_to_end"]["p95_ms"]
            if profile["end_to_end"]["p95_ms"] is not None
            and base["end_to_end"]["p95_ms"] is not None
            else None
        )
        rtf_delta = (
            profile["real_time_factor"]["median"] - base["real_time_factor"]["median"]
            if profile["real_time_factor"]["median"] is not None
            and base["real_time_factor"]["median"] is not None
            else None
        )
        deltas[label] = {
            "end_to_end_p95_delta_ms": end_delta,
            "real_time_factor_median_delta": rtf_delta,
            "stage_p95_delta_ms": stage_delta,
        }
    final = final or baseline
    if final in profiles:
        status = profiles[final]["end_to_end"]["status"]
        p95 = profiles[final]["end_to_end"]["p95_ms"]
        budget = profiles[final]["end_to_end"]["budget_ms"]
        if status == "PASS":
            reason = f"{final}: capture-to-apply p95 {p95:.1f} ms <= {budget:g} ms"
        elif status == "FAIL":
            reason = f"{final}: capture-to-apply p95 {p95:.1f} ms exceeds {budget:g} ms"
        else:
            reason = f"{final}: " + "; ".join(profiles[final]["end_to_end"]["evidence_issues"])
    else:
        status, reason = _INSUFFICIENT, f"No profile labelled {final!r}"
    return {
        "baseline": baseline,
        "final": final,
        "deltas": deltas,
        "gate": {
            "configuration": final,
            "status": status,
            "decision": "ADMIT" if status == "PASS" else "BLOCKED",
            "reason": reason,
        },
    }
