"""Known-answer and failure tests; unittest also runs these without pytest."""
import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from evaluation.metrics import (
    aggregate,
    evaluate_run,
    first_sustained,
    percentile,
    stopped_margin,
    validate_s0_criteria,
)
from evaluation.run_log import RunLogger, write_json
from navigation_control import Waypoint
from navigation_control.path import ReferencePath
from sim_harness.s0_acceptance import run_s0

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "configs/acceptance.json").read_text())
GOLDEN = json.loads((ROOT / "tests/fixtures/acceptance/golden.json").read_text())


def synthetic(directory):
    config = dict(CONFIG, dt_s=0.1)
    write_json(directory / "manifest.json", {"schema_version": 2, "seed": 1,
               "config": {"scenario": "S0", "acceptance": config, "evidence_kind": "known-answer fixture"}})
    rows, events = [], []
    for i in range(70):
        final = i >= 50
        target = 0 if final or 30 <= i <= 40 else 5
        speed = 0 if final else 4.5
        rows.append({"frame_id": i, "sim_time_s": i / 10,
                     "host_receive_s": i / 10, "simulation_frame_age_s": 0,
                     "host_applied_s": i / 10 + .02, "ego": {"speed_mps": speed},
                     "goal_distance_m": 0.5 if final else 10,
                     "planned_final_stop": final,
                     "diagnostics": {"target_speed_mps": target, "behaviour": "GOAL_STOP" if final else "CRUISE",
                                     "speed_caps_mps": {}, "signed_cte_m": .2, "observation_valid": True}})
        collision = [{"other_actor_id": 7}] if i in (0, 4, 10) else []
        events.append({"frame_id": i, "collisions": collision, "lane_invasions": [{}] if i == 2 else []})
    save_rows(directory, rows, events)
    return rows, events


def save_rows(directory, rows, events):
    for name, data in (("ticks", rows), ("evaluation", events)):
        (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))


class AcceptanceTests(unittest.TestCase):
    def test_authoritative_criteria_drift_rejected(self):
        definitions = json.loads((ROOT / "tests/fixtures/acceptance_definitions.json").read_text())
        validate_s0_criteria(CONFIG, definitions)
        changed = copy.deepcopy(definitions)
        changed["metrics"][0]["required_runs"] += 1
        with self.assertRaises(ValueError):
            validate_s0_criteria(CONFIG, changed)
        with self.assertRaises(ValueError):
            validate_s0_criteria(dict(CONFIG, mean_cte_m=.5), definitions)

    def test_geometry_and_stop_golden(self):
        g = GOLDEN
        path = ReferencePath(Waypoint(*p) for p in g["route"])
        projected = path.project(*g["point"])
        self.assertEqual(projected.progress_m, g["projection_progress_m"])
        self.assertEqual(projected.distance_m, g["projection_distance_m"])
        rows = [{"sim_time_s": i * .1, "frame_id": i, "ego": {"speed_mps": v}, "truth_front_gap_m": 4.1 - i * .1} for i, v in enumerate(g["speeds"])]
        c = dict(CONFIG, dt_s=.1)
        stop = first_sustained(rows, lambda r: r["ego"]["speed_mps"] <= .1, 1, .1)
        self.assertEqual(stop["frame_id"], g["stop_frame"])
        self.assertAlmostEqual(stopped_margin(rows, c), g["stopped_margin_m"])
        self.assertIsNone(stopped_margin(rows[:-1], c))
        self.assertEqual(percentile(g["percentile_input"], .95), g["p95"])

    def test_interrupted_stop_and_time_gap(self):
        rows = [{"sim_time_s": i * .1, "ego": {"speed_mps": 0}} for i in range(12)]
        rows[5]["ego"]["speed_mps"] = .11
        self.assertIsNone(stopped_margin(rows, dict(CONFIG, dt_s=.1)))
        self.assertIsNone(first_sustained([rows[0], rows[-1]], lambda r: True, 1, .1))

    def test_known_metrics_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            synthetic(p)
            result = evaluate_run(p)
            m = result["metrics"]
            self.assertTrue(m["route_complete"])
            for key, golden_key in (("false_stops", "false_stops"), ("collisions", "collision_count"), ("lane_invasions", "lane_invasion_count")):
                self.assertEqual(m[key], GOLDEN[golden_key])
            self.assertAlmostEqual(m["cruise_rms_mps"], GOLDEN["cruise_rms_mps"])
            self.assertAlmostEqual(m["mean_cte_m"], .2)
            self.assertAlmostEqual(m["latency_p95_ms"], 20)
            self.assertEqual(result["status"], "FAIL")

    def test_no_cruise_is_blocked_and_motion_is_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows, events = synthetic(p)
            for r, e in zip(rows, events):
                r["diagnostics"]["behaviour"] = "GOAL_STOP"
                r["diagnostics"]["target_speed_mps"] = 0
                r["planned_final_stop"] = True
                e["collisions"], e["lane_invasions"] = [], []
            save_rows(p, rows, events)
            self.assertEqual(evaluate_run(p)["status"], "BLOCKED")
            for r in rows:
                r["ego"]["speed_mps"] = 1
            save_rows(p, rows, events)
            self.assertFalse(evaluate_run(p)["metrics"]["route_complete"])

    def test_incomplete_and_clock_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows, events = synthetic(p)
            save_rows(p, rows, events[:-1])
            with self.assertRaises(ValueError):
                evaluate_run(p)
            rows[0]["host_applied_s"] = -1
            save_rows(p, rows, events)
            with self.assertRaises(ValueError):
                evaluate_run(p)

    def test_application_boundary_and_invalid_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows, events = synthetic(p)
            for r, e in zip(rows, events):
                e["collisions"], e["lane_invasions"] = [], []
                r["planned_final_stop"] = True
            rows[0]["diagnostics"]["observation_valid"] = False
            save_rows(p, rows, events)
            result = evaluate_run(p)
            self.assertFalse(result["checks"]["observation_valid"])
            self.assertEqual(result["status"], "FAIL")
            self.assertAlmostEqual(result["metrics"]["latency_p95_ms"], 20)
            rows[0]["host_applied_s"] = float("nan")
            save_rows(p, rows, events)
            with self.assertRaises(ValueError):
                evaluate_run(p)

    def test_five_offline_trials_repeatable_without_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("socket.socket.connect", side_effect=AssertionError("network forbidden")):
            directories = [Path(tmp) / str(i) for i in range(5)]
            for i, directory in enumerate(directories):
                self.assertEqual(run_s0(directory, CONFIG, i, ROOT)["status"], "PASS")
            self.assertEqual(aggregate(directories)["status"], "PASS")
            self.assertEqual(aggregate(directories[:4])["status"], "BLOCKED")
            self.assertEqual(aggregate([directories[0]] * 5)["status"], "BLOCKED")
            repeat = Path(tmp) / "repeat"
            run_s0(repeat, CONFIG, 0, ROOT)
            def deterministic(path):
                rows = [json.loads(s) for s in (path / "ticks.jsonl").read_text().splitlines()]
                for r in rows:
                    for k in ("host_receive_s", "host_applied_s"):
                        r.pop(k)
                    r["control"].pop("episode_id")
                    r["control"]["diagnostics"].pop("stage_durations_s")
                    r["diagnostics"].pop("timings_ms")
                return rows
            self.assertEqual(deterministic(directories[0]), deterministic(repeat))
            with self.assertRaises(FileExistsError):
                run_s0(repeat, CONFIG, 0, ROOT)

    def test_logger_serializes_dataclass_mappings_and_validates(self):
        # Capture one real harness frame without copying its image into the log.
        records = []
        original = RunLogger.record
        def capture(logger, *args, **kwargs):
            if not records:
                records.append((args, kwargs))
            return original(logger, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(RunLogger, "record", capture):
            run_s0(Path(tmp) / "source", CONFIG, 1, ROOT)
            args, kwargs = records[0]
            bundle, packet, diagnostics, evaluation = args
            with RunLogger(Path(tmp) / "log", config={}, seed=1, metadata={}, host_origin_s=bundle.received_monotonic_s) as logger:
                with self.assertRaises(ValueError):
                    logger.record(bundle, replace(packet, source_frame_id=999), diagnostics, evaluation, **kwargs)
                with self.assertRaises(ValueError):
                    logger.record(bundle, replace(packet, episode_id="foreign"), diagnostics, evaluation, **kwargs)
                with self.assertRaises(ValueError):
                    logger.record(*args, **dict(kwargs, goal_distance_m=math.nan))
                logger.record(*args, **kwargs)
                with self.assertRaises(ValueError):
                    logger.record(*args, **kwargs)
            row = json.loads((Path(tmp) / "log/ticks.jsonl").read_text())
            self.assertNotIn("rgb", row)
            self.assertEqual(row["host_receive_s"], 0)


if __name__ == "__main__":
    unittest.main()
