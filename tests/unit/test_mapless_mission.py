"""The baseline route and its S0–S5 scenarios on the mapless harness (``mapless_mission``),
and the G2 acceptance aggregator (``scripts/aggregate_mapless_acceptance.py``)."""

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from carla_adapter import mapless_mission as mm
from mapless.satnav import Command, LocalTangentPlane

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("aggregate_mapless_acceptance", ROOT / "scripts/aggregate_mapless_acceptance.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)
spec_rescore = importlib.util.spec_from_file_location("rescore_mission", ROOT / "scripts/rescore_mission.py")
rescore = importlib.util.module_from_spec(spec_rescore)
spec_rescore.loader.exec_module(rescore)


def bent_site(obstacle_s_m=None, junctions=((90.0, 110.0, 468),)):
    """A 200 m chain: 60 m straight, a 45 degree bend, straight on; a junction at 90–110 m."""
    pts = [np.array([0.0, 0.0])]
    heading = 0.0
    for k in range(1, 201):
        if 60 <= k < 90:
            heading += math.radians(45.0) / 30
        pts.append(pts[-1] + np.array([math.cos(heading), math.sin(heading)]))
    pts = np.array(pts)
    arcs = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    return mm.MissionSite("test_chain", pts, arcs, tuple(junctions), 110.0, 3.5, 1.75, "sidewalk", [], ("1:0:1", "468:0:-1", "5:0:-1"), obstacle_s_m)


def test_the_site_places_points_along_the_chain_with_a_lateral_offset():
    site = bent_site()
    x, y, h = site.at(10.0)
    assert (x, y, h) == pytest.approx((10.0, 0.0, 0.0))
    x, y, h = site.at(10.0, 4.5)
    assert (x, y) == pytest.approx((10.0, 4.5))  # lateral is to the left
    x, y, h = site.at(150.0)
    assert h == pytest.approx(math.radians(45.0), abs=1e-6) and site.length_m == pytest.approx(200.0)
    assert site.at(1e9)[0] == pytest.approx(site.points_xy[-1, 0])


def test_the_obstacle_goes_where_the_baseline_puts_it():
    scenarios = {sid: mm.load_scenario(sid, ROOT / "configs/scenarios") for sid in mm.SCENARIO_IDS}
    assert mm.obstacle_placement(bent_site(), scenarios["S0"]) is None
    s, lateral, bp = mm.obstacle_placement(bent_site(), scenarios["S1"])
    assert (s, lateral, bp) == (40.0, 0.0, "vehicle.tesla.model3")  # the scenario's own s
    s, lateral, bp = mm.obstacle_placement(bent_site(obstacle_s_m=120.0), scenarios["S1"])
    assert s == 120.0  # the route definition's placement wins (a1_test01: mid-way along its straight)
    s, lateral, bp = mm.obstacle_placement(bent_site(), scenarios["S2"])
    assert bp == "walker.pedestrian.0001"
    s, lateral, bp = mm.obstacle_placement(bent_site(), scenarios["S3"])
    assert lateral == 4.5
    s, _, _ = mm.obstacle_placement(bent_site(obstacle_s_m=120.0), scenarios["S5"])
    assert s == 116.0  # after the turn exit, whatever the route says
    with pytest.raises(ValueError):
        mm.obstacle_placement(bent_site(obstacle_s_m=190.0), scenarios["S4"])
    with pytest.raises(ValueError):
        mm.load_scenario("S9", ROOT / "configs/scenarios")


def test_the_route_is_sparse_with_a_straight_at_the_junction(monkeypatch):
    plane = LocalTangentPlane(-33.9, 151.2)
    carla_map = SimpleNamespace(transform_to_geolocation=lambda loc: SimpleNamespace(latitude=plane.to_geodetic(loc.x, -loc.y)[0], longitude=plane.to_geodetic(loc.x, -loc.y)[1]))
    monkeypatch.setitem(sys.modules, "carla", SimpleNamespace(Location=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)))
    site = bent_site()
    route = mm.mission_route(carla_map, site)
    commands = [p.command for p in route.points]
    assert commands.count(Command.STRAIGHT) == 1 and commands[-1] is Command.LANEFOLLOW
    local = [plane.to_local(p.latitude_deg, p.longitude_deg) for p in route.points]
    s_of = [float(site.arcs[np.argmin(np.hypot(*(site.points_xy - np.array(xy)).T))]) for xy in local]
    entry = s_of[commands.index(Command.STRAIGHT)]
    assert entry == pytest.approx(90.0, abs=1.0) and s_of[commands.index(Command.STRAIGHT) + 1] == pytest.approx(115.0, abs=1.0)
    assert not any(80.0 <= s < 90.0 for s, c in zip(s_of, commands) if c is Command.LANEFOLLOW)  # no ordinary point crowds the entry
    assert s_of[-1] == pytest.approx(200.0, abs=1.0) and all(b - a > 5.0 for a, b in zip(s_of, s_of[1:]))
    assert route.provenance["junctions"][0]["junction_id"] == 468
    # Without a junction: plain points every 25 m to the goal.
    plain = mm.mission_route(carla_map, bent_site(junctions=()))
    assert all(p.command is Command.LANEFOLLOW for p in plain.points) and len(plain.points) == 9


def test_the_chain_is_resolved_with_its_junction_span_and_kerb(monkeypatch):
    """``resolve_mission`` on a stand-in map: three lanes (approach, junction 468, exit),
    the junction's entry and exit found where its lane begins and the exit lane begins."""
    import carla_adapter.topology as topology_mod  # noqa: PLC0415
    import planning.mission as mission_mod  # noqa: PLC0415

    def lane(key, x0, x1, junction=None):
        return SimpleNamespace(segment_id=key, junction_id=junction, points_xyz_m=tuple((float(x), 1.75, 0.0) for x in range(x0, x1 + 1)))

    lanes = [lane("1:0:1", 0, 90), lane("468:0:-1", 90, 110, junction=468), lane("5:0:-1", 110, 200)]
    points = tuple(p for ln in lanes for p in ln.points_xyz_m[:-1]) + (lanes[-1].points_xyz_m[-1],)
    mission = SimpleNamespace(segment_ids=tuple(ln.segment_id for ln in lanes), points_xyz_m=points, turn_exit_s_m=110.0, map_validation=dict(route_definition=dict(obstacle_s_m=120.0)))
    monkeypatch.setattr(topology_mod, "export_lane_topology", lambda carla_map, sampling_distance_m, navigation_speed_cap_mps: SimpleNamespace(lanes=lanes))
    monkeypatch.setattr(mission_mod, "load_route_definition", lambda path: dict(name="stand_in"))
    monkeypatch.setattr(mission_mod, "mission_from_definition", lambda topology, definition: mission)
    wp = SimpleNamespace(lane_width=3.5, get_right_lane=lambda: SimpleNamespace(lane_type="Sidewalk", lane_width=2.0, get_right_lane=lambda: None))
    carla = SimpleNamespace(Location=lambda x, y, z: SimpleNamespace(x=x, y=y, z=z), LaneType=SimpleNamespace(Driving="Driving", Sidewalk="Sidewalk"))
    carla_map = SimpleNamespace(get_waypoint=lambda loc, project_to_road, lane_type: wp)
    site = mm.resolve_mission(carla, carla_map, "unused.json")
    assert site.name == "stand_in" and site.length_m == pytest.approx(200.0) and site.lane_width_m == 3.5
    assert site.junctions == ((90.0, 110.0, 468),) and site.turn_exit_s_m == 110.0 and site.obstacle_s_m == 120.0
    assert site.kerb_distance_m == pytest.approx(1.75) and site.kerb_source == "sidewalk" and site.segment_ids == ("1:0:1", "468:0:-1", "5:0:-1")


def _rows(stops, n=400, removed_after=None, obstacle_s=40.0):
    """A run: 2 m/s to ``stops`` (a hold there with the gap), then on to the goal at 90 m."""
    rows, s, t, held, removed = [], 0.0, 0.0, 0, None
    for k in range(n):
        mode, reason, v = "driving", "", 2.0
        gap = None if obstacle_s is None or removed is not None else obstacle_s - 1.2 - (s + 3.78)
        if stops is not None and s >= stops and (removed is None or t - removed < 1.0):
            mode, reason, v = "stopping", "holding for obstacle (blocked)" if removed is None else "holding for obstacle_hold (clear)", 0.0
            held += 1
            if removed is None and removed_after is not None and held * 0.05 >= removed_after:
                removed = t  # a second to move off follows
        rows.append(dict(time_s=t, mode=mode, reason=reason, truth=dict(speed_mps=v, s_m=s, gap_m=gap, goal_distance_m=max(0.0, 90.0 - s))))
        s += v * 0.05
        t += 0.05
    return rows, removed


def test_the_scenario_gates_are_scored_as_the_plan_says():
    s1 = mm.load_scenario("S1", ROOT / "configs/scenarios")
    s4 = mm.load_scenario("S4", ROOT / "configs/scenarios")
    s0 = mm.load_scenario("S0", ROOT / "configs/scenarios")
    s3 = mm.load_scenario("S3", ROOT / "configs/scenarios")
    # S1: held 3 m short, the car removed after 2 s, the goal reached.
    log, removed = _rows(stops=31.5, removed_after=2.0)
    log[-1]["truth"]["goal_distance_m"] = 0.5
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s1, log, "finished", 0, removed, 0.05)}
    assert rows["obstacle_stop"][0] and 1.5 <= rows["obstacle_stop"][1]["gap_m"] <= 5.0 and rows["obstacle_stop"][1]["within_plan_band"]
    assert rows["resume"][0] and rows["resume"][1]["moved_off_s"] == pytest.approx(1.05, abs=0.1)
    # S1 held but never removed (the run ended first): resume failed; never held: not judged.
    log, _ = _rows(stops=31.5, removed_after=None, n=400)
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s1, log, "time limit", 0, None, 0.05)}
    assert rows["resume"][0] is False and rows["obstacle_stop"][0]
    log, _ = _rows(stops=31.5, removed_after=None, n=100)
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s1, log, "time limit", 0, None, 0.05)}
    assert rows["resume"][0] is None and rows["obstacle_stop"][0] is False
    # S4: the hold is the end.
    log, _ = _rows(stops=31.5, n=600)
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s4, log, "held at the obstacle", 60, None, 0.05)}
    assert rows["held"][0] and rows["obstacle_stop"][0] and "resume" not in rows
    # Too close: the band fails it.
    log, _ = _rows(stops=34.0, n=600)
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s4, log, "held at the obstacle", 60, None, 0.05)}
    assert rows["obstacle_stop"][0] is False
    # Thirteenth review, 2 (reproduced there): a stop that began inside the band at 1.55 m
    # and crept to 1.35 m at a standstill passed on its first gap. The whole hold is judged.
    log, _ = _rows(stops=31.5, n=600)
    held = [r for r in log if r["mode"] == "stopping"]
    for k, r in enumerate(held):
        r["truth"]["gap_m"] = max(1.35, 1.55 - 0.002 * k)
        r["truth"]["speed_mps"] = 0.04
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s4, log, "held at the obstacle", 60, None, 0.05)}
    values = rows["obstacle_stop"][1]
    assert rows["obstacle_stop"][0] is False and values["gap_m"] == 1.55 and values["gap_min_m"] == 1.35 and values["gap_creep_m"] == 0.2 and 0.05 < values["inside_band_share"] < 0.15  # 25 of the ~270 held ticks were still inside the band
    # S0: the goal. S3: the goal and no stop for the roadside car.
    log, _ = _rows(stops=None, obstacle_s=None, n=1000)
    log[-1]["truth"]["goal_distance_m"] = 0.3
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s0, log, "finished", 0, None, 0.05)}
    assert rows == {"goal": (True, dict(end="finished", goal_distance_m=0.3))}
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s3, log, "finished", 0, None, 0.05)}
    assert rows["goal"][0] and rows["no_stop_for_roadside_car"][0]
    log, _ = _rows(stops=31.5, n=600)
    rows = {name: (ok, v) for name, ok, v in mm.score_mission(s3, log, "held at the obstacle", 60, None, 0.05)}
    assert rows["no_stop_for_roadside_car"][0] is False and rows["goal"][0] is False


def _report(folder, scenario, seed, status, failed=(), commit="abc", mission="a1_test01", route_noise_m=2.0, suite="suite0", stack=True):
    folder.mkdir(parents=True, exist_ok=True)
    checks = [dict(check="no_collision", status="PASS")] + [dict(check=c, status="FAIL") for c in failed]
    report = dict(acceptance=dict(scenario=scenario, seed=seed, mission=mission, route_noise_m=route_noise_m), status=status, end="finished", source_commit=commit, suite_sha256=suite, checks=checks, collisions=[])
    if stack:
        report["stack"] = dict(admission="B", segmenter="constant", segmenter_sha256=None, stack_config_sha256="stack0")
    (folder / "drive_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_the_aggregator_passes_only_a_complete_homogeneous_set(tmp_path):
    seeds = (5308, 5309, 5310, 5311, 5312)
    for sid in agg.SCENARIOS:
        for seed in seeds:
            _report(tmp_path / f"{sid}_{seed}", sid, seed, "MEASURED_PASS_OVER_BUDGET" if sid == "S2" and seed == 5308 else "MEASURED_PASS")
    summary = agg.aggregate(agg.collect(tmp_path))
    assert summary["verdict"] == "MEASURED_PASS" and summary["runs"] == 30 and "1 of 30" in summary["note"]
    assert "verdict: MEASURED_PASS" in agg.table(summary)
    # One run short: INCOMPLETE. One failed gate: MEASURED_FAIL. Another commit: INCOMPLETE.
    short = agg.aggregate([r for r in agg.collect(tmp_path) if not (r["scenario"] == "S5" and r["seed"] == 5312)])
    assert short["verdict"] == "INCOMPLETE"
    _report(tmp_path / "extra", "S3", 5313, "MEASURED_FAIL", failed=("no_stop_for_roadside_car",))
    failed = agg.aggregate(agg.collect(tmp_path))
    assert failed["verdict"] == "MEASURED_FAIL" and failed["scenarios"]["S3"]["failed"][0]["checks"] == ["no_stop_for_roadside_car"]
    assert "FAIL S3 seed 5313" in agg.table(failed)
    (tmp_path / "extra" / "drive_report.json").unlink()
    # A run that passed its gates without reaching what its scenario is for is counted apart.
    _report(tmp_path / "idle", "S1", 5313, "NOT_EXERCISED")
    idle = agg.aggregate(agg.collect(tmp_path))
    assert idle["verdict"] == "INCOMPLETE" and len(idle["scenarios"]["S1"]["not_exercised"]) == 1 and not idle["scenarios"]["S1"]["incomplete"]
    assert "NOT_EXERCISED S1 seed 5313" in agg.table(idle)
    (tmp_path / "idle" / "drive_report.json").unlink()
    (tmp_path / "idle").rmdir()
    (tmp_path / "extra").rmdir()
    _report(tmp_path / "other_commit", "S0", 5399, "MEASURED_PASS", commit="def")
    mixed = agg.aggregate(agg.collect(tmp_path))
    assert mixed["verdict"] == "INCOMPLETE" and not mixed["homogeneous"] and mixed["scenarios"]["S0"]["unexpected_seeds"] == [5399]
    assert "SEEDS S0: missing [], unexpected [5399]" in agg.table(mixed)
    (tmp_path / "other_commit" / "drive_report.json").unlink()
    (tmp_path / "other_commit").rmdir()
    # Twelfth review, 4 (reproduced there): a run without a commit was "the same commit", and
    # mixed route noise passed. A missing identity field, or two identities, is INCOMPLETE.
    _report(tmp_path / "S0_5308", "S0", 5308, "MEASURED_PASS", commit=None)
    blank = agg.aggregate(agg.collect(tmp_path))
    assert blank["verdict"] == "INCOMPLETE" and blank["unprovenanced"][0]["missing"] == ["code"] and "NO PROVENANCE" in agg.table(blank)
    _report(tmp_path / "S0_5308", "S0", 5308, "MEASURED_PASS", route_noise_m=0.0)
    noise = agg.aggregate(agg.collect(tmp_path))
    assert noise["verdict"] == "INCOMPLETE" and not noise["homogeneous"] and noise["identities"] == 2 and noise["identity"] is None
    _report(tmp_path / "S0_5308", "S0", 5308, "MEASURED_PASS", stack=False)
    assert agg.aggregate(agg.collect(tmp_path))["unprovenanced"][0]["missing"] == ["stack_config_sha256", "admission"]
    _report(tmp_path / "S0_5308", "S0", 5308, "MEASURED_PASS")
    whole = agg.aggregate(agg.collect(tmp_path))
    assert whole["verdict"] == "MEASURED_PASS" and whole["identity"]["route_noise_m"] == 2.0 and whole["seeds_required"] == [5308, 5309, 5310, 5311, 5312]
    # The frozen seed set is required exactly; "any" five distinct seeds is the loose form.
    assert agg.aggregate(agg.collect(tmp_path), seeds=(1, 2, 3, 4, 5))["verdict"] == "INCOMPLETE"
    assert agg.aggregate(agg.collect(tmp_path), seeds=None)["verdict"] == "MEASURED_PASS"
    # A report that is not a mission run is ignored; an unreadable one is listed and blocks the verdict.
    (tmp_path / "lane").mkdir()
    (tmp_path / "lane" / "drive_report.json").write_text(json.dumps(dict(status="MEASURED_PASS", checks=[])), encoding="utf-8")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "drive_report.json").write_text("{", encoding="utf-8")
    runs = agg.collect(tmp_path)
    assert sum("error" in r for r in runs) == 1 and sum("error" not in r for r in runs) == 30
    torn = agg.aggregate(runs)
    assert torn["verdict"] == "INCOMPLETE" and len(torn["unreadable"]) == 1 and "UNREADABLE" in agg.table(torn)
    assert agg.main([str(tmp_path), "--output", str(tmp_path / "out.json")]) == 1 and (tmp_path / "out.json").exists()


def test_rescoring_a_recorded_run_rejudges_the_hold_and_keeps_the_original(tmp_path):
    """Thirteenth review, 2: a batch scored before the whole-hold rule can be re-scored from
    its logs; the driving code stays ``source_commit``, the scoring code ``rescored.commit``,
    and the report as the run wrote it is kept beside the new one."""
    s4 = mm.load_scenario("S4", ROOT / "configs/scenarios")
    log, _ = _rows(stops=31.5, n=600)
    held = [r for r in log if r["mode"] == "stopping"]
    for k, r in enumerate(held):
        r["truth"]["gap_m"] = max(1.35, 1.55 - 0.002 * k)
        r["truth"]["speed_mps"] = 0.04
    folder = tmp_path / "S4_5308"
    folder.mkdir()
    (folder / "drive_log.jsonl").write_text("\n".join(json.dumps(r) for r in log), encoding="utf-8")
    # The run's own report, scored on the first gap: a pass.
    first = {name: (ok, v) for name, ok, v in mm.score_mission(s4, log, "held at the obstacle", len(held), None, 0.05)}
    checks = [dict(check="no_collision", status="PASS"), dict(check="obstacle_stop", status="PASS", gap_m=1.55, held_s=first["obstacle_stop"][1]["held_s"]), dict(check="held", status="PASS", end="held at the obstacle", held_s=first["held"][1]["held_s"])]
    (folder / "drive_report.json").write_text(json.dumps(dict(acceptance=dict(scenario="S4", seed=5308, mission="a1_test01", route_noise_m=2.0), status="MEASURED_PASS", end="held at the obstacle", source_commit="abc", checks=checks)), encoding="utf-8")
    assert rescore.rescore(folder, commit="def") == ("MEASURED_PASS", "MEASURED_FAIL")
    report = json.loads((folder / "drive_report.json").read_text(encoding="utf-8"))
    by_name = {c["check"]: c for c in report["checks"]}
    assert report["status"] == "MEASURED_FAIL" and report["source_commit"] == "abc" and report["rescored"]["commit"] == "def" and report["rescored"]["previous_status"] == "MEASURED_PASS"
    assert by_name["obstacle_stop"]["status"] == "FAIL" and by_name["obstacle_stop"]["gap_min_m"] == 1.35 and by_name["held"]["status"] == "PASS" and by_name["no_collision"]["status"] == "PASS"
    original = json.loads((folder / "drive_report.original.json").read_text(encoding="utf-8"))
    assert original["status"] == "MEASURED_PASS" and "rescored" not in original
    # Re-scoring again changes nothing and keeps the first original.
    assert rescore.rescore(folder, commit="ghi") == ("MEASURED_FAIL", "MEASURED_FAIL")
    assert json.loads((folder / "drive_report.original.json").read_text(encoding="utf-8"))["status"] == "MEASURED_PASS"
    assert rescore.main([str(tmp_path)]) == 0
