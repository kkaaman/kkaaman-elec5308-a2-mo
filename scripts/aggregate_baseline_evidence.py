"""Build the TEST-08 review report from an explicit, append-only evidence register.

This inventories owner-reviewed measurements; it does not evaluate raw CARLA logs
or grant baseline admission. Source evaluators and reviewers retain that duty.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = tuple(f"S{i}" for i in range(6))
FAULTS = ("hold_occlusion", "hold_missing_depth", "hold_track_expiry",
          "hold_id_switch", "controller_invalid_state", "controller_behind_target",
          "controller_path_loss")


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def artifact_errors(artifacts, base):
    errors = []
    if not artifacts:
        return ["No retained evidence artifacts"]
    for item in artifacts:
        path = (base / item["path"]).resolve()
        if not path.is_file():
            errors.append(f"Missing artifact: {item['path']}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"Artifact SHA-256 mismatch: {item['path']}")
    return errors


def measurement(value, definition):
    if not definition["applicable"]:
        return "N/A"
    if not isinstance(value, dict):
        return "MISSING"
    number = value.get("value")
    count = value.get("n")
    if (type(number) not in (int, float) or not math.isfinite(number)
            or type(count) is not int or count < 1 or not value.get("method")
            or not value.get("evidence")):
        return "MISSING"
    if definition["unit"] == "count" and (number < 0 or number != int(number)):
        return "FAIL"
    if definition["unit"] == "fraction" and not 0 <= number <= 1:
        return "FAIL"
    rule = definition["rule"]
    if rule == "report":
        return "REPORTED"
    passed = {">=": lambda: number >= definition["lower"],
              "<=": lambda: number <= definition["upper"],
              "=": lambda: number == definition["lower"]}[rule]()
    return "MEETS" if passed else "FAIL"


def aggregate(register, definitions, base):
    if register.get("schema") != 1:
        raise ValueError("Expected register schema 1")
    attempts = register["attempts"]
    ids = [a["run_id"] for a in attempts]
    if len(ids) != len(set(ids)) or any(not x for x in ids):
        raise ValueError("Run IDs must be nonempty and unique; retain retries separately")
    criteria = {s: [m for m in definitions["metrics"] if m["scenario"] == s]
                for s in SCENARIOS}
    results, failures, summaries = [], [], []
    for a in attempts:
        scenario = a["scenario"]
        if scenario not in SCENARIOS or type(a["trial"]) is not int or not 1 <= a["trial"] <= 5:
            raise ValueError("Each attempt needs scenario S0-S5 and trial 1-5")
        errors = artifact_errors(a.get("artifacts"), base)
        for key in ("candidate_commit", "configuration_sha256"):
            if not register.get(key) or a.get(key) != register[key]:
                errors.append(f"Missing or different {key}")
        if a.get("environment") != "live_carla" or a.get("complete_stack") is not True:
            errors.append("Not live CARLA complete-stack acceptance evidence")
        if not a.get("reviewer") or not a.get("review_record"):
            errors.append("Owner/reviewer evidence review pending")
        if a.get("outcome") not in ("completed", "failed", "aborted"):
            raise ValueError("Attempt outcome must be completed, failed or aborted")
        if type(a.get("seed")) is not int:
            errors.append("Run seed missing")
        elif any(other is not a and other["scenario"] == scenario
                 and other["trial"] != a["trial"] and other.get("seed") == a["seed"]
                 for other in attempts):
            errors.append("Seed reused across distinct required trials")
        checks = {m["metric"]: measurement(a.get("metrics", {}).get(m["metric"]), m)
                  for m in criteria[scenario]}
        artifact_paths = {item["path"] for item in a.get("artifacts", [])}
        for name, metric in a.get("metrics", {}).items():
            if isinstance(metric, dict) and metric.get("evidence") not in artifact_paths:
                errors.append(f"Metric evidence not in hashed artifacts: {name}")
        # A numeric fraction alone is never proof of endpoint-safe completion.
        if scenario != "S4" and a.get("safe_completion_verified") is not True:
            errors.append("Safe endpoint completion unverified (route progress is insufficient)")
        if a["outcome"] != "completed" or "FAIL" in checks.values():
            state = "FAIL"
        elif errors or "MISSING" in checks.values():
            state = "BLOCKED"
        else:
            state = "READY_FOR_REVIEW"
        reasons = errors + [f"{k}: {v}" for k, v in checks.items() if v in ("FAIL", "MISSING")]
        if a["outcome"] != "completed":
            reasons.append(f"Attempt {a['outcome']}: {a.get('failure_reason') or 'reason missing'}")
        result = {"run_id": a["run_id"], "scenario": scenario, "trial": a["trial"],
                  "state": state, "reasons": reasons, "checks": checks, "record": a}
        results.append(result)
        if state != "READY_FOR_REVIEW":
            failures.append({k: result[k] for k in ("run_id", "scenario", "trial", "state", "reasons")})
    for s in SCENARIOS:
        rows = [r for r in results if r["scenario"] == s]
        slots = {r["trial"] for r in rows}
        missing = sorted(set(range(1, 6)) - slots)
        metric_summary = []
        for m in criteria[s]:
            values = [r["record"]["metrics"][m["metric"]]["value"] for r in rows
                      if r["checks"][m["metric"]] in ("MEETS", "REPORTED", "FAIL")
                      and isinstance(r["record"].get("metrics", {}).get(m["metric"]), dict)
                      and type(r["record"]["metrics"][m["metric"]].get("value")) in (int, float)
                      and math.isfinite(r["record"]["metrics"][m["metric"]]["value"])]
            metric_summary.append({"metric": m["metric"], "unit": m["unit"],
                                   "rule": m["rule"], "lower": m["lower"], "upper": m["upper"],
                                   "applicable": m["applicable"], "n_attempts": len(values),
                                   "min": min(values) if values else None,
                                   "max": max(values) if values else None,
                                   "median": statistics.median(values) if values else None,
                                   "sample_stdev": statistics.stdev(values) if len(values) > 1 else None})
        summaries.append({"scenario": s, "required_trials": 5, "attempts": len(rows),
                          "missing_trials": missing, "failed": sum(r["state"] == "FAIL" for r in rows),
                          "blocked": sum(r["state"] == "BLOCKED" for r in rows),
                          "metrics": metric_summary})
    faults = []
    for name in FAULTS:
        rows = [f for f in register["fault_tests"] if f["test"] == name]
        ready = bool(rows) and all(f.get("outcome") == "meets" and f.get("reviewer")
                                  and f.get("review_record") and f.get("observations")
                                  and f.get("candidate_commit") == register.get("candidate_commit")
                                  and bool(register.get("candidate_commit"))
                                  and not artifact_errors(f.get("artifacts"), base) for f in rows)
        faults.append({"test": name, "state": "READY_FOR_REVIEW" if ready else "BLOCKED",
                       "attempts": rows})
    unknown = set(f["test"] for f in register["fault_tests"]) - set(FAULTS)
    if unknown:
        raise ValueError(f"Unknown fault tests: {sorted(unknown)}")
    complete = all(not s["missing_trials"] and not s["failed"] and not s["blocked"]
                   for s in summaries) and all(f["state"] == "READY_FOR_REVIEW" for f in faults)
    return {"state": "READY_FOR_REVIEW" if complete else "BLOCKED", "scenarios": summaries,
            "attempts": results, "failure_log": failures, "fault_tests": faults,
            "limitation": "Reviewed register aggregation only; no raw-log evaluation or admission."}


def markdown(report):
    lines = ["# TEST-08 baseline evidence review", "", f"State: **{report['state']}**", "",
             report["limitation"], "", "## Thirty required trial slots", "",
             "| Scenario | Trial | Attempts and state |", "|---|---|---|"]
    for s in SCENARIOS:
        for trial in range(1, 6):
            rows = [r for r in report["attempts"] if r["scenario"] == s and r["trial"] == trial]
            text = "; ".join(f"{r['run_id']}: {r['state']}" for r in rows) or "NOT RUN / evidence pending"
            lines.append(f"| {s} | {trial} | {text} |")
    lines += ["", "## Metric coverage and descriptive uncertainty", "",
              "All attempts, including failures, contribute. Retries are not independent trials.",
              "Sample standard deviation is descriptive spread, not a confidence interval."]
    for s in report["scenarios"]:
        lines += ["", f"### {s['scenario']}", "",
                  "| Metric | Rule | n attempts | Min | Max | Median | Sample SD |",
                  "|---|---|---:|---|---|---|---|"]
        for m in s["metrics"]:
            rule = "N/A" if not m["applicable"] else ("report only" if m["rule"] == "report" else f"{m['rule']} {m['lower'] if m['lower'] is not None else m['upper']} {m['unit']}")
            cells = [str(m[k]) if m[k] is not None else "pending" for k in ("min", "max", "median", "sample_stdev")]
            lines.append(f"| {m['metric']} | {rule} | {m['n_attempts']} | " + " | ".join(cells) + " |")
    lines += ["", "## Failure and insufficient-evidence log", ""]
    for r in report["failure_log"]:
        lines.append(f"- {r['scenario']} trial {r['trial']}, {r['run_id']}: {r['state']}; " + "; ".join(r["reasons"]))
    if not report["failure_log"]:
        lines.append("No attempt records supplied; this is not a zero-failure result."
                     if not report["attempts"] else "No recorded failing/blocked attempts; review remains required.")
    lines += ["", "## Fault evidence", ""]
    for f in report["fault_tests"]:
        lines.append(f"- {f['test']}: {f['state']} ({len(f['attempts'])} retained attempts)")
        for attempt in f["attempts"]:
            lines.append(f"  - {attempt.get('run_id', 'ID missing')}: {attempt.get('outcome')}; {attempt.get('observations', 'observations missing')}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("register", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    report = aggregate(read_json(args.register), read_json(ROOT / "tests/fixtures/acceptance_definitions.json"), args.register.resolve().parent)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "baseline_table.md").write_text(markdown(report), encoding="utf-8")
    print(report["state"])
    return 0 if report["state"] == "READY_FOR_REVIEW" else 1


if __name__ == "__main__":
    raise SystemExit(main())
