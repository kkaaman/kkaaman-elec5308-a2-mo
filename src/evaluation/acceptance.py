"""Recompute evidence; keep every attempt and distinguish simulation scope."""

import json
from pathlib import Path

from evaluation.metrics import evaluate, load_run, summarize
from evaluation.run_logger import file_sha256


def aggregate(folders, criteria, *, required_repeats=5, navigation_only=False):
    records, errors = [], []
    for folder in folders:
        folder = Path(folder)
        try:
            manifest, rows, events = load_run(folder)
            summary = summarize(rows, events, manifest)
            records.append(
                dict(
                    folder=str(folder),
                    manifest=manifest,
                    summary=summary,
                    manifest_sha256=file_sha256(folder / "manifest.json"),
                )
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(dict(folder=str(folder), error=str(exc)))
    for item in records:
        m, s = item["manifest"], item["summary"]
        if m["scenario"] == "S3":
            matches = [
                x
                for x in records
                if x["manifest"]["scenario"] == "S0"
                and all(
                    x["manifest"].get(k) == m.get(k)
                    for k in (
                        "seed",
                        "engine",
                        "mode",
                        "config_sha256",
                        "mission_sha256",
                        "source_sha256",
                        "model_sha256",
                    )
                )
            ]
            if len(matches) == 1:
                baseline = matches[0]["summary"]["metrics"].get("cruise_mean_mps")
                roadside = s["metrics"].get("cruise_mean_mps")
                if baseline is not None and roadside is not None:
                    s["metrics"]["cruise_drop_mps"] = baseline - roadside
        item["acceptance"] = evaluate(
            s, criteria, m["scenario"], collected=m["status"] == "COLLECTED"
        )
    scenarios = ("S0",) if navigation_only else tuple(f"S{i}" for i in range(6))
    counts = {s: sum(x["manifest"]["scenario"] == s for x in records) for s in scenarios}
    homogeneous = all(
        len({x["manifest"].get(k) for x in records}) == 1
        for k in (
            "engine",
            "config_sha256",
            "mission_sha256",
            "source_sha256",
            "model_sha256",
            "mode",
        )
    )
    distinct = all(
        len({x["manifest"]["seed"] for x in records if x["manifest"]["scenario"] == s}) == counts[s]
        for s in scenarios
    )
    coverage = (
        all(n == required_repeats for n in counts.values())
        and len(records) == len(scenarios) * required_repeats
    )
    measured_pass = (
        not errors
        and coverage
        and distinct
        and homogeneous
        and all(x["acceptance"]["status"] == "PASS" for x in records)
    )
    live = bool(records) and all(x["manifest"]["engine"] == "carla" for x in records)
    frozen = bool(records) and all(
        x["manifest"].get("configuration_status") == "FROZEN_FOR_VALIDATION" for x in records
    )
    correct_mode = all(
        x["manifest"].get("mode") == ("navigation" if navigation_only else "full_stack")
        for x in records
    )
    accepted = measured_pass and live and correct_mode and (frozen or navigation_only)
    return dict(
        status="MEASURED_PASS_REVIEW_REQUIRED" if accepted else "NOT_ACCEPTED",
        measured_metrics_pass=measured_pass,
        live_carla=live,
        configuration_frozen=frozen,
        coverage=coverage,
        unique_seeds=distinct,
        consistent_provenance=homogeneous,
        mode="navigation" if navigation_only else "full_stack",
        counts=counts,
        required_repeats=required_repeats,
        errors=errors,
        runs=records,
        scope="Harness results cannot close a CARLA gate; collection alone is not acceptance.",
    )


def find_runs(root):
    return sorted(path.parent for path in Path(root).glob("*/manifest.json"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument(
        "--criteria",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "tests/fixtures/acceptance_definitions.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navigation-only", action="store_true")
    args = parser.parse_args()
    report = aggregate(
        find_runs(args.runs),
        json.loads(args.criteria.read_text()),
        navigation_only=args.navigation_only,
    )
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"{report['status']} | {report['counts']} | {args.output}")
    return 0 if report["status"] == "MEASURED_PASS_REVIEW_REQUIRED" else 2
