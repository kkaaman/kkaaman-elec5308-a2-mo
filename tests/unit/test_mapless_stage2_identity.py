"""Stage 2, object identity: regressions for the review of 3e9083a (H1, H4, M1, M2, M6) and
the external review of the full review of 73ae759.

Tracker:
- a neighbour's radar return cannot set the lead's speed (3-D gate with the radial speed);
- the radar keeps informing a lead after its radar and LiDAR tracks merge;
- neither a merge nor a radar-held track's first LiDAR fix makes a parked car move;
- tracks moving in opposite directions are not merged, nor a walker into a radar-held car;
- spread-out or disagreeing Doppler rejections are not a manoeuvre;
- a static radar-only track where the LiDAR saw nothing is not clutter.

Splitting: a pedestrian, a cyclist or a car's rear face beside a structure stays an object.

Stack: a walker beside the lead stays a conflict candidate, the lead's own pieces do not;
a vehicle's run to the path is blocked only by a persistent wall no track covers.
"""

import math

import numpy as np

from mapless.contracts import ObjectTrack, TrackStatus
from mapless.detections import DetectionsConfig, cluster_and_split
from mapless.ego_motion import Pose2D
from mapless.prediction import PredictionConfig, motion_state, predict
from mapless.radar import RadarMeasurement
from mapless.stack import StackConfig, outside_lead_body, road_reachable
from mapless.terrain import ClassifiedSweep, PointClass
from mapless.tracking import LidarMeasurement, Tracker, TrackerConfig

ORIGIN = Pose2D(0.0, 0.0, 0.0)
SENSOR = np.array([2.5, 0.0])
DT = 0.05


def ret(xy, radial):
    """A radar object in the vehicle frame; ``radial``: its own radial speed."""
    xy = np.asarray(xy, dtype=float)
    rel = xy - SENSOR
    return RadarMeasurement("radar_front", xy, float(np.hypot(*rel)), float(math.atan2(rel[1], rel[0])), float(radial), 3, SENSOR)


def radial_of(xy, v):
    rel = np.asarray(xy, float) - SENSOR
    return float(np.asarray(v, float) @ (rel / np.hypot(*rel)))


def lid(xy, extent=(4.5, 1.8, 0.0)):
    xy = np.asarray(xy, dtype=float)
    return LidarMeasurement(xy, extent, None, 50, 0.3 + 0.01 * float(np.hypot(*xy)))


# -- tracker -------------------------------------------------------------------------
def test_a_neighbours_radar_return_cannot_set_the_leads_speed():
    for neighbour_speed in (0.0, 10.0):  # a parked car's corner; a car overtaking alongside
        tr = Tracker(TrackerConfig())
        for k in range(60):
            lead = np.array([25.0 + 5.0 * k * DT, 0.0])
            tracks = tr.update(k * DT, ORIGIN, lidar=[lid(lead)], radar=[ret(lead, radial_of(lead, (5.0, 0.0)))])
        for k in range(60, 100):
            lead = np.array([25.0 + 5.0 * k * DT, 0.0])
            other = lead + (0.0, 2.6)
            radar = [ret(other, radial_of(other, (neighbour_speed, 0.0)))] if k % 10 == 0 else []
            tracks = tr.update(k * DT, ORIGIN, lidar=[lid(lead)], radar=radar)
            mine = min(tracks, key=lambda t: np.hypot(*(t.position - lead)))
            assert abs(mine.velocity[0] - 5.0) < 1.0, f"neighbour {neighbour_speed}: t={k * DT:.2f} vx {mine.velocity[0]:.2f}"


def test_a_neighbour_cannot_capture_a_radar_only_lead_while_its_own_return_is_missing():
    """Beyond the LiDAR's reach: a car 3.3 m to the side of a 5 m/s radar-only lead, at
    7 m/s, while the lead's own return is missing for four ticks. Its returns agree with
    each other, but sit to one side: no manoeuvre (review of 3e9083a, H4)."""
    tr = Tracker(TrackerConfig())
    for k in range(60):
        lead = np.array([36.0 + 5.0 * k * DT, 0.0])
        tracks = tr.update(k * DT, ORIGIN, radar=[ret(lead, radial_of(lead, (5.0, 0.0)))])
    lead_id = tracks[0].track_id
    for k in range(60, 70):
        t = k * DT
        lead = np.array([36.0 + 5.0 * t, 0.0])
        other = np.array([lead[0] - 0.3 + 2.0 * (t - 3.0), 3.3])
        radar = [ret(other, radial_of(other, (7.0, 0.0)))] + ([ret(lead, radial_of(lead, (5.0, 0.0)))] if k >= 64 else [])
        tracks = tr.update(t, ORIGIN, radar=radar)
        mine = next(trk for trk in tracks if trk.track_id == lead_id)
        assert abs(mine.velocity[0] - 5.0) < 0.3 and abs(mine.position[1]) < 0.3, (k, mine.velocity, mine.position)
    assert len(tracks) == 2  # the neighbour has its own


def test_the_radar_still_informs_a_merged_lead():
    tr = Tracker(TrackerConfig())
    for k in range(40):  # radar only, 36 m out
        tr.update(k * DT, ORIGIN, radar=[ret((36.0 + 2.0 * k * DT, 0.0), 2.0)])
    for k in range(40, 80):  # the LiDAR clusters it 2.5 m nearer than the radar's point
        x = 36.0 + 2.0 * k * DT
        tracks = tr.update(k * DT, ORIGIN, lidar=[lid((x - 2.5, 0.0))], radar=[ret((x, 0.0), 2.0)])
    assert len(tracks) == 1 and set(tracks[0].support) == {"lidar", "radar"}
    speeds = []
    for k in range(80, 86):  # stopped dead
        tracks = tr.update(k * DT, ORIGIN, lidar=[lid((x - 2.5, 0.0))], radar=[ret((x, 0.0), 0.0)])
        speeds.append(tracks[0].velocity[0])
    assert speeds[2] < 0.6, np.round(speeds, 2)


def test_neither_a_merge_nor_a_first_lidar_fix_makes_a_parked_car_move():
    """A parked car the radar holds by a return off its front (1.8 m nearer than its
    centre), until the LiDAR clusters it: once confirmed with LiDAR, it is static."""
    car = np.array([50.0, -2.5])
    tr = Tracker(TrackerConfig())
    pc = PredictionConfig()
    for k in range(160):
        t = k * DT
        pose = Pose2D(4.0 * t, 0.0, 0.0)
        body = pose.to_body(car[None])[0]
        lidar = [lid(body)] if body[0] < 29.0 else []
        tracks = tr.update(t, pose, lidar=lidar, radar=[ret(pose.to_body((car - (1.8, 0.0))[None])[0], 0.0)])
        for trk in tracks:
            if "lidar" in trk.support and trk.status in (TrackStatus.CONFIRMED, TrackStatus.COASTING):
                assert motion_state(trk, pc) == "static", f"t={t:.2f} recent={trk.recent_speed_mps} travelled={trk.travelled_m}"
                assert trk.travelled_m < 0.75


def test_tracks_moving_in_opposite_directions_are_not_merged():
    """A lead at +5 m/s the LiDAR holds; an oncoming car at -5 m/s only the radar sees
    passes through its footprint: two objects (external review: they merged into one with
    the oncoming car's velocity)."""
    tr = Tracker(TrackerConfig())
    merged = 0
    for k in range(120):
        t = k * DT
        lead = np.array([20.0 + 5.0 * t, 0.0])
        oncoming = np.array([70.0 - 5.0 * t, 0.6])
        tracks = tr.update(t, ORIGIN, lidar=[lid(lead)], radar=[ret(oncoming, radial_of(oncoming, (-5.0, 0.0)))])
        merged += tr.radar_stats.get("merged", 0)
        mine = min(tracks, key=lambda trk: np.hypot(*(trk.position - lead)))
        if k > 20:
            assert mine.velocity[0] > 3.0, f"t={t:.2f}: lead vx {mine.velocity[0]:.2f}"
    assert merged == 0


def test_a_walker_beside_a_radar_held_car_keeps_its_own_track():
    tr = Tracker(TrackerConfig())
    for k in range(30):  # a car 35 m out at 8 m/s, radar only
        tr.update(k * DT, ORIGIN, radar=[ret((35.0 + 8.0 * k * DT, 0.0), 8.0)])
    for k in range(30, 50):  # a walker-sized cluster the LiDAR sees 0.8 m beside the car's radar point
        x = 35.0 + 8.0 * k * DT
        tracks = tr.update(k * DT, ORIGIN, lidar=[lid((x, 0.8), extent=(0.5, 0.5, 0.0))], radar=[ret((x, 0.0), 8.0)])
        assert not tr.radar_stats.get("merged")
    assert len(tracks) == 2


def test_spread_out_or_disagreeing_doppler_rejections_are_not_a_manoeuvre():
    car = lid((20.0, 0.0))

    def settled():
        tr = Tracker(TrackerConfig())
        for k in range(20):
            tr.update(k * DT, ORIGIN, lidar=[car])
        return tr

    tr = settled()  # the same reading 3 m/s away, but 0.2 s apart (the run expires after 0.3 s)
    for k in range(20, 40):
        tracks = tr.update(k * DT, ORIGIN, lidar=[car], radar=[ret((20.0, 0.0), 2.5)] if k % 4 == 0 else [])
        assert not tr.radar_stats.get("doppler_manoeuvres")
    assert abs(tracks[0].velocity[0]) < 0.3
    tr = settled()  # every tick, one-sided, but not agreeing with each other
    for k, rate in zip(range(20, 30), (1.5, 2.8, 1.5, 2.8, 1.5, 2.8, 1.5, 2.8, 1.5, 2.8)):
        tracks = tr.update(k * DT, ORIGIN, lidar=[car], radar=[ret((20.0, 0.0), rate)])
        assert not tr.radar_stats.get("doppler_manoeuvres")
    assert abs(tracks[0].velocity[0]) < 0.3
    tr = settled()  # every tick and agreeing: a manoeuvre by the third
    for k in range(20, 23):
        tracks = tr.update(k * DT, ORIGIN, lidar=[car], radar=[ret((20.0, 0.0), 1.7)])
    assert tr.radar_stats.get("doppler_manoeuvres") and tracks[0].velocity[0] > 0.8


def test_a_static_radar_only_track_where_the_lidar_saw_nothing_is_not_clutter():
    for seen, kept in ((False, True), (True, False)):
        tr = Tracker(TrackerConfig())
        for k in range(120):  # carried in from 40 m by the car driving at 4 m/s
            t = k * DT
            pose = Pose2D(4.0 * t, 0.0, 0.0)
            tracks = tr.update(t, pose, radar=[ret((40.0 - 4.0 * t, 5.0), 0.0)], lidar_view=lambda xy, seen=seen: np.full(len(xy), seen))
        assert bool(tracks) is kept


# -- splitting ------------------------------------------------------------------------
RNG = np.random.default_rng(0)


def _sweep(*parts):
    xyz = np.vstack(parts)
    bounds = np.cumsum([0] + [len(p) for p in parts])
    return ClassifiedSweep(xyz, xyz[:, 2], np.full(len(xyz), PointClass.OBSTACLE), np.zeros(1), np.zeros(1)), bounds


def _facing_arc(cx, cy, rx, ry, n, z0=0.2, z1=1.7):
    facing = math.atan2(-cy, -cx)
    a = facing + RNG.uniform(-math.pi / 2, math.pi / 2, n)
    return np.column_stack((cx + rx * np.cos(a), cy + ry * np.sin(a), RNG.uniform(z0, z1, n)))


def _line(x0, x1, y0, y1, step=0.05, z=0.9):
    n = int(math.hypot(x1 - x0, y1 - y0) / step)
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((x0 + t * (x1 - x0), y0 + t * (y1 - y0), np.full(n, z)))


def _object_share(parts, user):
    sweep, bounds = _sweep(*parts)
    objects, _ = cluster_and_split(sweep, DetectionsConfig())
    members = set(range(bounds[user], bounds[user + 1]))
    return sum(len(members & set(map(int, g))) for g in objects) / len(members)


def test_road_users_beside_a_structure_stay_objects():
    wall = _line(-5.0, 40.0, -5.0, -5.0)
    rear = np.column_stack((RNG.uniform(25.0, 25.25, 120), RNG.uniform(-0.9, 0.9, 120), RNG.uniform(0.4, 1.4, 120)))
    cars = [np.column_stack((RNG.uniform(x0, x0 + 4.5, 150), -2.6 + RNG.normal(0.0, 0.03, 150), RNG.uniform(0.3, 1.4, 150))) for x0 in (8.0, 13.5, 19.0)]
    cases = {
        "pedestrian 0.7 m from a wall at 10 m": ([wall, _facing_arc(10.0, -4.3, 0.25, 0.15, 40)], 1),
        "pedestrian 1.2 m from a wall at 25 m": ([wall, _facing_arc(25.0, -3.8, 0.25, 0.15, 25)], 1),
        "cyclist 0.9 m from a fence at 20 m": ([_line(-5.0, 60.0, -3.2, -3.2), _facing_arc(20.0, -2.0, 0.15, 0.3, 30, 0.3, 1.8)], 1),
        "pedestrian stepping out between parked cars": ([*cars, _facing_arc(13.0, -2.0, 0.25, 0.15, 30)], 3),
        "a lead's rear face, a railing 1.1 m off its side": ([_line(-5.0, 60.0, -2.0, -2.0), rear], 1),
    }
    for name, (parts, user) in cases.items():
        assert _object_share(parts, user) == 1.0, name
    # The structure itself stays structure: a railing 65 m long, and a 4.9 m stretch of one.
    sweep, _ = _sweep(_line(-5.0, 60.0, -2.0, -2.0))
    objects, structures = cluster_and_split(sweep, DetectionsConfig())
    assert objects == [] and structures


# -- stack ------------------------------------------------------------------------------
U = {k: 0.2 for k in ("car", "truck", "bike", "pedestrian", "unknown")}


def track(tid, x, y, vx, vy, length, width, classes=U):
    return ObjectTrack(tid, 0.0, np.array([x, y, vx, vy]), np.diag([0.05] * 4), classes, TrackStatus.CONFIRMED, 20, 0, support=("lidar",), footprint=(length, width, 0.0), travelled_m=3.0, recent_speed_mps=max(0.1, math.hypot(vx, vy)))


def test_a_walker_beside_the_lead_stays_a_candidate_but_the_leads_pieces_do_not():
    c = StackConfig()
    path = np.column_stack((np.arange(-2.0, 60.0, 0.5), np.zeros(124)))
    lead = track(1, 15.0, 0.0, 3.0, 0.0, 4.5, 1.8)
    piece = track(2, 16.0, 0.6, 0.0, 1.5, 0.4, 0.4)  # a fragment of its roof, jittering sideways
    walker = track(3, 16.0, 1.6, 0.0, 0.0, 0.5, 0.5)  # walked up beside it from the pavement
    tracks = [lead, piece, walker]
    history = {}
    s_car = 2.0  # the path starts 2 m behind the rear axle

    def kept(walker_y):
        tracks[2] = track(3, 16.0, walker_y, 0.0, -1.2, 0.5, 0.5)
        preds = [predict(t, c.prediction) for t in tracks]
        out = outside_lead_body(tracks, preds, tracks[0], 15.0, ORIGIN, path, s_car, c, followed=True, history=history, dt_s=DT)
        return sorted(p.track_id for p in out)

    for k in range(12):  # 0.6 s on the pavement beside the lane, 4 m out
        assert kept(4.0) == [1, 3]
    assert kept(1.6) == [1, 3]  # now beside the lead, inside its body margin: still a road user
    # A track born inside the body and never seen outside it is a piece; the lead stays.
    history.clear()
    assert kept(1.6) == [1]
    # ... unless the camera says it is a pedestrian.
    tracks[1] = track(2, 16.0, 0.6, 0.0, 1.5, 0.4, 0.4, classes={**U, "pedestrian": 0.9, "car": 0.025, "truck": 0.025, "bike": 0.025, "unknown": 0.025})
    assert 2 in kept(1.6)


def test_a_vehicles_run_to_the_path_is_blocked_only_by_a_persistent_uncovered_wall():
    c = StackConfig()
    path = np.column_stack((np.arange(-2.0, 60.0, 0.5), np.zeros(124)))
    car = track(5, 20.0, -15.0, 0.0, 4.0, 4.5, 1.9)  # heading for the path from 15 m right

    class Grid:
        def __init__(self, value, half_width=50.0):
            self.value, self.half_width = value, half_width

        def tall_evidence(self, xy):
            xy = np.asarray(xy, dtype=float)
            band = (xy[:, 1] < -9.5) & (xy[:, 1] > -10.5) & (np.abs(xy[:, 0] - 20.0) <= self.half_width)
            return np.where(band, self.value, 0.0).astype(np.float32)

    def dropped(grid, tracks=None, history=None, ticks=1):
        tracks = tracks or [car]
        out = None
        for _ in range(ticks):
            _, out = road_reachable(tracks, [predict(car, c.prediction)], path, grid, 5.7, c, history=history)
        return out

    assert dropped(Grid(0.0)) == []  # a kerb or pavement: no tall evidence at all
    assert dropped(Grid(8.0)) == []  # another crossing car's fading trail
    assert dropped(Grid(50.0, half_width=0.2)) == []  # a post: a car fits past it
    parked = track(6, 20.0, -10.0, 0.0, 0.0, 4.5, 1.9)  # the tall cells are a parked car's (a track)
    assert dropped(Grid(50.0), tracks=[car, parked]) == []
    assert dropped(Grid(50.0)) == [5]  # a railing across its way
    history = {}
    assert dropped(Grid(50.0), history=history, ticks=2) == []  # not yet persistent
    assert dropped(Grid(50.0), history=history) == [5]
