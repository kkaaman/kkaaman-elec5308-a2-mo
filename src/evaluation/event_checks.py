"""Collision/lane-invasion counting and declared legal crossings (BL-B-07).

The run metrics and the live sensor verification both call these functions,
so the verification exercises the same rules that score acceptance runs.
Evaluation-only: nothing here is imported by the autonomy stack.
"""

import json
import math
from pathlib import Path

CONTACT_SPLIT_S = 1.0
GAP_TOLERANCE_M = 0.05


def count_contacts(events, split_s=CONTACT_SPLIT_S):
    """Unique contacts: callbacks for one actor merge until a gap >= split_s."""
    contacts, last = 0, {}
    for event in sorted(
        (e for e in events if e.get("type") == "collision"), key=lambda e: e["sim_time_s"]
    ):
        key = event["other_actor_id"]
        if key not in last or event["sim_time_s"] - last[key] >= split_s:
            contacts += 1
        last[key] = event["sim_time_s"]
    return contacts


def count_unplanned_invasions(events):
    """Every lane-marking crossing counts unless explicitly declared legal."""
    return sum(
        1 for e in events if e.get("type") == "lane_invasion" and not e.get("declared_legal", False)
    )


def load_declarations(path):
    """Read predeclared legal crossings; an absent file declares none.

    Format: {"schema_version": 1, "crossings": [{"from_s_m": a, "to_s_m": b,
    "markings": ["Broken", ...], "reason": "..."}]}. Route arc length is the
    rear-axle mission s. Junctions are never exempted wholesale.
    """
    if path is None:
        return ()
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1:
        raise ValueError("Unsupported crossing declaration schema")
    result = []
    for row in document.get("crossings", ()):
        low, high = float(row["from_s_m"]), float(row["to_s_m"])
        markings = tuple(str(m) for m in row["markings"])
        if not (math.isfinite(low) and math.isfinite(high)) or high <= low or not markings:
            raise ValueError("Declared crossing needs a finite s interval and marking types")
        if not str(row.get("reason", "")).strip():
            raise ValueError("Declared crossing needs a reason")
        result.append(dict(from_s_m=low, to_s_m=high, markings=markings, reason=row["reason"]))
    return tuple(result)


def annotate_invasion(event, route_s_m, declarations):
    """Return a copy marked legal only if every crossed marking is declared here."""
    row = dict(event, route_s_m=route_s_m)
    markings = tuple(row.get("markings", ()))
    legal = bool(markings) and any(
        d["from_s_m"] <= route_s_m <= d["to_s_m"] and all(m in d["markings"] for m in markings)
        for d in declarations
    )
    row["declared_legal"] = legal
    return row


def check_gap(expected_m, measured_m, tolerance_m=GAP_TOLERANCE_M):
    if measured_m is None or not math.isfinite(measured_m):
        return dict(status="FAIL", reason="no_measured_gap", expected_m=expected_m)
    error = measured_m - expected_m
    return dict(
        status="PASS" if abs(error) <= tolerance_m else "FAIL",
        expected_m=expected_m,
        measured_m=measured_m,
        error_m=error,
        tolerance_m=tolerance_m,
    )


def verification_status(gap_checks, collision_check, lane_check):
    """MEASURED_PASS only when every component produced passing live evidence."""
    parts = [*gap_checks, collision_check, lane_check]
    if not gap_checks or any(p is None for p in parts):
        return "INCOMPLETE"
    return "MEASURED_PASS" if all(p["status"] == "PASS" for p in parts) else "MEASURED_FAIL"
