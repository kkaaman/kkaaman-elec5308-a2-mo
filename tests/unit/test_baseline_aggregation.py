"""TEST-08 aggregation must not turn partial or synthetic evidence into acceptance."""
import copy
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("aggregation", ROOT / "scripts/aggregate_baseline_evidence.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
DEFINITIONS = MODULE.read_json(ROOT / "tests/fixtures/acceptance_definitions.json")


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        (self.base / "raw.txt").write_text("retained evidence")
        self.artifacts = [{"path": "raw.txt", "sha256": hashlib.sha256(b"retained evidence").hexdigest()}]
        self.register = {"schema": 1, "candidate_commit": "a" * 40,
                         "configuration_sha256": "b" * 64, "attempts": [], "fault_tests": []}

    def attempt(self, scenario="S0", trial=1):
        metrics = {}
        for m in DEFINITIONS["metrics"]:
            if m["scenario"] == scenario and m["applicable"]:
                value = m["lower"] if m["lower"] is not None else m["upper"]
                metrics[m["metric"]] = {"value": value if value is not None else 1,
                                        "n": 100, "method": "owner documented windows",
                                        "evidence": "raw.txt"}
        return {"run_id": f"{scenario}-{trial}", "scenario": scenario, "trial": trial, "seed": 5307 + trial,
                "candidate_commit": self.register["candidate_commit"],
                "configuration_sha256": self.register["configuration_sha256"],
                "environment": "live_carla", "complete_stack": True, "reviewer": "owner",
                "review_record": "review-1", "safe_completion_verified": True,
                "outcome": "completed", "artifacts": self.artifacts, "metrics": metrics}

    def report(self):
        return MODULE.aggregate(self.register, DEFINITIONS, self.base)

    def test_empty_covers_thirty_slots_and_every_metric_without_results(self):
        report = self.report()
        self.assertEqual(report["state"], "BLOCKED")
        self.assertEqual(sum(len(s["missing_trials"]) for s in report["scenarios"]), 30)
        self.assertEqual(sum(len(s["metrics"]) for s in report["scenarios"]), 55)
        self.assertIn("not a zero-failure result", MODULE.markdown(report))

    def test_fraction_does_not_prove_safe_completion(self):
        row = self.attempt()
        row["safe_completion_verified"] = False
        self.register["attempts"] = [row]
        self.assertEqual(self.report()["attempts"][0]["state"], "BLOCKED")

    def test_synthetic_results_and_missing_artifacts_block(self):
        row = self.attempt()
        row["environment"] = "offline"
        self.register["attempts"] = [row]
        self.assertEqual(self.report()["attempts"][0]["state"], "BLOCKED")
        row["environment"] = "live_carla"
        (self.base / "raw.txt").write_text("changed")
        self.assertIn("SHA-256 mismatch", str(self.report()["failure_log"]))

    def test_failed_retry_is_never_discarded(self):
        failed = self.attempt()
        failed["outcome"] = "failed"
        retry = copy.deepcopy(failed)
        retry.update(run_id="retry", outcome="completed")
        self.register["attempts"] = [failed, retry]
        report = self.report()
        self.assertEqual(len(report["attempts"]), 2)
        self.assertEqual(report["scenarios"][0]["failed"], 1)
        self.assertEqual(report["state"], "BLOCKED")

    def test_missing_report_only_value_still_blocks(self):
        row = self.attempt("S1")
        del row["metrics"]["Resume delay (median)"]
        self.register["attempts"] = [row]
        self.assertEqual(self.report()["attempts"][0]["state"], "BLOCKED")

    def test_s4_route_is_not_applicable(self):
        row = self.attempt("S4")
        row["safe_completion_verified"] = False
        self.register["attempts"] = [row]
        result = self.report()["attempts"][0]
        self.assertEqual(result["state"], "READY_FOR_REVIEW")
        self.assertEqual(result["checks"]["Route completion (minimum per run)"], "N/A")

    def test_nonfinite_boolean_and_zero_samples_cannot_meet(self):
        m = DEFINITIONS["metrics"][0]
        for value, n in [(float("nan"), 10), (True, 10), (1, 0)]:
            self.assertEqual(MODULE.measurement({"value": value, "n": n,
                                               "method": "x", "evidence": "x"}, m), "MISSING")

    def test_threshold_failure_cannot_be_hidden_by_other_trials(self):
        self.register["attempts"] = [self.attempt("S0", t) for t in range(1, 6)]
        self.register["attempts"][0]["metrics"]["Cross-track mean (worst run)"]["value"] = 0.31
        report = self.report()
        self.assertEqual(report["scenarios"][0]["failed"], 1)
        self.assertEqual(report["scenarios"][0]["metrics"][3]["max"], 0.31)

    def test_duplicate_trial_seeds_block_distinct_slots(self):
        a, b = self.attempt("S0", 1), self.attempt("S0", 2)
        b["seed"] = a["seed"]
        self.register["attempts"] = [a, b]
        self.assertTrue(all(r["state"] == "BLOCKED" for r in self.report()["attempts"]))

    def test_unhashed_metric_reference_blocks(self):
        row = self.attempt()
        row["metrics"]["Cross-track mean (worst run)"]["evidence"] = "unretained.json"
        self.register["attempts"] = [row]
        self.assertEqual(self.report()["attempts"][0]["state"], "BLOCKED")

    def test_duplicate_run_ids_rejected(self):
        self.register["attempts"] = [self.attempt(), self.attempt()]
        with self.assertRaises(ValueError):
            self.report()

    def test_complete_register_requires_fault_evidence_and_never_grants_pass(self):
        self.register["attempts"] = [self.attempt(s, t) for s in MODULE.SCENARIOS for t in range(1, 6)]
        self.assertEqual(self.report()["state"], "BLOCKED")
        self.register["fault_tests"] = [dict(test=f, run_id=f"fault-{f}", outcome="meets", reviewer="owner",
                                             configuration_sha256=self.register["configuration_sha256"],
                                             environment="live_carla", complete_stack=True,
                                             review_record="review", observations="latched stop checked",
                                             candidate_commit=self.register["candidate_commit"],
                                             artifacts=self.artifacts) for f in MODULE.FAULTS]
        self.assertEqual(self.report()["state"], "READY_FOR_REVIEW")


    def ready_register(self):
        self.register["attempts"] = [self.attempt(s, t) for s in MODULE.SCENARIOS for t in range(1, 6)]
        self.register["fault_tests"] = [dict(test=f,run_id=f"fault-{f}",outcome="meets",
            reviewer="owner",review_record="review",observations="latched stop checked",
            candidate_commit=self.register["candidate_commit"],
            configuration_sha256=self.register["configuration_sha256"],
            environment="live_carla",complete_stack=True,artifacts=self.artifacts) for f in MODULE.FAULTS]

    def test_fault_missing_or_different_configuration_blocks(self):
        for value in (None, "c"*64):
            self.ready_register()
            self.register["fault_tests"][0]["configuration_sha256"] = value
            self.assertEqual(self.report()["state"], "BLOCKED")

    def test_fault_missing_id_or_nonlive_stack_blocks(self):
        for field, value in (("run_id", None), ("environment", "offline"), ("complete_stack", False)):
            self.ready_register()
            self.register["fault_tests"][0][field] = value
            self.assertEqual(self.report()["state"], "BLOCKED")

    def test_fault_run_ids_must_be_unique_and_not_alias_scenario_runs(self):
        for alias_scenario in (False, True):
            self.ready_register()
            self.register["fault_tests"][1]["run_id"] = (self.register["attempts"][0]["run_id"]
                if alias_scenario else self.register["fault_tests"][0]["run_id"])
            with self.assertRaisesRegex(ValueError, "unique"):
                self.report()

    def test_duplicate_json_fields_are_not_silently_overwritten(self):
        path = self.base / "ambiguous.json"
        path.write_text('{"schema": 1, "schema": 2}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            MODULE.read_json(path)


if __name__ == "__main__":
    unittest.main()
