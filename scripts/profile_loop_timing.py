"""TEST-22 / TEST-17: loop-timing profile from retained run diagnostics.

Compare labelled configurations (for example headless versus HUD) from S0 run
folders (``samples.csv``) or warm-model ``timing.jsonl`` records, and judge the
final configuration against the frozen 100 ms capture-to-apply p95 budget.

Example::

    python scripts/profile_loop_timing.py --output runs/TEST-22/profile ^
        --run headless=runs/S0/headless-20261101/run-01 ^
        --run hud=runs/S0/hud-20261101/run-01 ^
        --run warm=runs/BL-C-05/governance/timing.jsonl ^
        --baseline headless --final hud

Writes ``latency_profile.json`` and ``latency_profile.png``. Exit code 0 only
when the final configuration's gate is PASS; 1 otherwise (FAIL or insufficient).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from evaluation.loop_timing import (  # noqa: E402
    DEFAULT_BUDGET_MS,
    compare_profiles,
    load_timing_rows,
    profile_rows,
)


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("Run label must be nonempty")
    return label.strip(), Path(path)


def plot(profiles, rows_by_label, budget_ms, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    for label, rows in rows_by_label.items():
        key = profiles[label]["end_to_end"]["source_key"]
        if key is None:
            continue
        scale = 1000.0 if key.endswith("_s") else 1.0
        values = sorted(
            row[key] * scale
            for row in rows
            if isinstance(row.get(key), (int, float)) and row[key] >= 0
        )
        if values:
            ax.plot(
                values,
                [(i + 1) / len(values) for i in range(len(values))],
                label=f"{label} (n={len(values)}, p95={profiles[label]['end_to_end']['p95_ms']:.1f} ms)",
            )
    ax.axvline(budget_ms, color="red", linestyle="--", linewidth=1, label=f"{budget_ms:g} ms budget")
    ax.set(xlabel="End-to-end wall latency (ms)", ylabel="Fraction", title="Capture-to-apply CDF")
    ax.legend(fontsize=7)
    ax = axes[1]
    stages = sorted({name for profile in profiles.values() for name in profile["stages"]})
    width = 0.8 / max(1, len(profiles))
    for offset, (label, profile) in enumerate(profiles.items()):
        values = [profile["stages"].get(name, {}).get("p95_ms") or 0.0 for name in stages]
        positions = [i + offset * width for i in range(len(stages))]
        ax.bar(positions, values, width=width, label=label)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(stages))])
    ax.set_xticklabels(stages, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="p95 (ms)", title="Per-stage p95 wall cost")
    if stages:
        ax.legend(fontsize=7)
    fig.suptitle("Loop timing profile (measurements; gate verdict in latency_profile.json)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=parse_run, action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--baseline", help="Label the others are compared against (default: first run)")
    parser.add_argument("--final", help="Label judged against the budget (default: baseline)")
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    labels = [label for label, _ in args.run]
    if len(set(labels)) != len(labels):
        parser.error("Run labels must be unique")
    baseline = args.baseline or labels[0]
    if baseline not in labels:
        parser.error("--baseline must name one of the runs")
    if args.final and args.final not in labels:
        parser.error("--final must name one of the runs")
    args.output.mkdir(parents=True, exist_ok=False)
    profiles, rows_by_label, sources = {}, {}, {}
    for label, path in args.run:
        rows, kind = load_timing_rows(path)
        rows_by_label[label] = rows
        sources[label] = {"path": str(path), "kind": kind}
        profiles[label] = profile_rows(rows, label=label, budget_ms=args.budget_ms)
    comparison = compare_profiles(profiles, baseline=baseline, final=args.final)
    report = {"sources": sources, "profiles": profiles, "comparison": comparison}
    (args.output / "latency_profile.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot(profiles, rows_by_label, args.budget_ms, args.output / "latency_profile.png")
    gate = comparison["gate"]
    print(f"{gate['decision']} | {gate['reason']} | {args.output}")
    return 0 if gate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
