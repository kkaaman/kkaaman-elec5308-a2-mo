"""Independent geometry, temporal, behaviour and distance-budget checks."""

from dataclasses import replace

import numpy as np
import pytest

from autonomy.interfaces import EgoPose, FrameBundle
from geometry.camera import CameraIntrinsics
from navigation_control.types import VehicleState
from obstacle_perception.association import Footprint
from obstacle_perception.track_filter import TrackFilter, TrackFilterConfig
from obstacle_perception.types import ObservationMetadata
from obstacle_perception.visibility import DepthVisibility
from planning.behaviour import Behaviour, BehaviourState
from planning.corridor import RouteGeometry
from planning.speed_governor import SpeedGovernor, speed_for_distance, stopping_distance


def footprint(
    frame=0,
    time=0.0,
    identity=1,
    *,
    corners=((20, -1), (24, -1), (24, 1), (20, 1)),
    source="rgb",
    episode="e",
):
    return Footprint(
        ObservationMetadata(episode, frame, time, source, "v1"), "car", corners, identity, 0.2, 0.9
    )


@pytest.mark.parametrize(
    "corners,expected",
    [
        (((20, -1), (24, -1), (24, 1), (20, 1)), 16),
        (((20, 3), (24, 3), (24, 5), (20, 5)), None),
        (((-10, -1), (-6, -1), (-6, 1), (-10, 1)), None),
        (((20, -4), (24, -4), (24, 4), (20, 4)), 16),
    ],
)
def test_corridor_front_offset_once(corners, expected):
    route = RouteGeometry(((0, 0), (40, 0)))
    assert route.gap(footprint(corners=corners), 0, 4, 1.2) == expected


def test_corridor_follows_bend_instead_of_infinite_start_tangent():
    route = RouteGeometry(((0, 0), (10, 0), (10, 20)))
    obj = footprint(corners=((9, 8), (11, 8), (11, 10), (9, 10)))
    assert route.gap(obj, 0, 4, 1) == pytest.approx(14)
    assert route.project(9, 8).lateral_m == pytest.approx(1)
    assert route.project(2, -1).lateral_m == pytest.approx(-1)
    assert route.footprint_interval(((15, -1), (17, -1), (17, 1), (15, 1)), half_width_m=1) is None


def test_confirmation_loss_and_namespace_isolation():
    f = TrackFilter(TrackFilterConfig(2, 3, 0.3))
    f.reset("e")
    a = footprint()
    assert not f.update((a,), a.metadata)[0].confirmed
    b = footprint(1, 0.05)
    assert f.update((b,), b.metadata)[0].confirmed
    missing = replace(b.metadata, frame_id=2, sim_time_s=0.10)
    held = f.update((), missing)[0]
    assert held.confirmed and not held.observed_this_frame
    switched = footprint(3, 0.15, identity=2)
    rows = f.update((switched,), switched.metadata)
    assert len(rows) == 2 and not next(x for x in rows if x.track_id == 2).confirmed
    expired = replace(missing, frame_id=10, sim_time_s=1.0)
    assert f.update((), expired) == ()
    with pytest.raises(ValueError, match="episode"):
        f.update((), replace(expired, episode_id="other", frame_id=11, sim_time_s=2.0))


def test_anonymous_confirms_only_by_spatial_persistence():
    # Same place on consecutive frames: a persistent untracked object confirms.
    f = TrackFilter()
    f.reset("e")
    rows = []
    for i in range(2):
        a = footprint(i, i * 0.05, identity=None)
        rows = f.update((a,), a.metadata)
    assert len(rows) == 1 and rows[0].confirmed
    # Flicker at places further apart than the gate never confirms.
    f.reset("e")
    for i in range(4):
        x = 20 + 3 * i
        a = footprint(i, i * 0.05, identity=None, corners=((x, -1), (x + 2, -1), (x + 2, 1), (x, 1)))
        assert all(not row.confirmed for row in f.update((a,), a.metadata))
    # Another source never merges into an anonymous record of the first.
    f.reset("e")
    a = footprint(0, 0.0, identity=None)
    f.update((a,), a.metadata)
    b = footprint(1, 0.05, identity=None, source="other")
    assert all(not row.confirmed for row in f.update((b,), b.metadata))


def test_stale_and_duplicate_observations_rejected():
    f = TrackFilter()
    f.reset("e")
    a = footprint()
    f.update((a,), a.metadata)
    with pytest.raises(ValueError):
        f.update((a,), a.metadata)


def test_median_does_not_hide_sudden_nearer_surface():
    f = TrackFilter()
    f.reset("e")
    for i, x in enumerate((20, 20, 10)):
        a = footprint(i, i * 0.05, corners=((x, -1), (x + 2, -1), (x + 2, 1), (x, 1)))
        row = f.update((a,), a.metadata)[0]
    assert min(p[0] for p in row.corners_world_m) == 10


def test_stopping_budget_inverse_includes_delay():
    for v in (0, 2, 5, 8):
        assert speed_for_distance(stopping_distance(v, 1.5, 0.5), 1.5, 0.5) == pytest.approx(v)
    assert speed_for_distance(10, 1.5, 0.5) < speed_for_distance(10, 1.5, 0)


def test_governor_straight_curve_and_emergency_priority():
    g = SpeedGovernor()
    for _ in range(100):
        target, _ = g.limit(speed=5, dt=0.05, curvature=((0, 0),))
    assert target == pytest.approx(5)
    target, _ = g.limit(speed=5, dt=0.05, curvature=((0, 0.25),))
    assert target == pytest.approx(2)
    target, emergency = g.limit(speed=5, dt=0.05, obstacle_gap=2)
    assert emergency and target == 0
    target, _ = g.limit(speed=0, dt=0.05)
    assert target == pytest.approx(0.05)


def test_hold_does_not_resume_after_track_expiry_or_unknown_visibility():
    b = Behaviour()
    assert (
        b.update(time_s=0, speed=0, obstacle=True, must_stop=True, clear=False)
        == BehaviourState.HOLD
    )
    for i in range(1, 30):
        assert (
            b.update(time_s=i * 0.05, speed=0, obstacle=False, must_stop=False, clear=False)
            == BehaviourState.HOLD
        )
    for i in range(30, 40):
        state = b.update(time_s=i * 0.05, speed=0, obstacle=False, must_stop=False, clear=True)
    assert state in {BehaviourState.RESUME, BehaviourState.CRUISE}
    assert (
        b.update(time_s=2.1, speed=1, obstacle=True, must_stop=True, clear=False)
        == BehaviourState.STOPPING
    )


def test_fault_and_finished_latch_until_reset():
    b = Behaviour()
    b.fault("missing frame")
    assert (
        b.update(time_s=1, speed=0, obstacle=False, must_stop=False, clear=True)
        == BehaviourState.FAULT_STOP
    )
    b.reset()
    assert (
        b.update(time_s=0, speed=0, obstacle=False, must_stop=False, clear=True, finished=True)
        == BehaviourState.FINISHED
    )
    assert (
        b.update(time_s=1, speed=0, obstacle=False, must_stop=False, clear=True)
        == BehaviourState.FINISHED
    )


def test_depth_clearance_requires_visible_fresh_pixels_not_blank_detections():
    cal = CameraIntrinsics.from_horizontal_fov(100, 60, np.pi / 2).calibration(
        ((0, 0, 1, 0), (-1, 0, 0, 0), (0, -1, 0, 2), (0, 0, 0, 1))
    )
    frame = FrameBundle(
        "e",
        1,
        0.05,
        1,
        EgoPose(VehicleState(0, 0, 0, 0)),
        np.zeros((60, 100, 3), np.uint8),
        cal,
        np.full((60, 100), 50, np.float32),
        1,
    )
    obj = footprint(corners=((10, -1), (12, -1), (12, 1), (10, 1)))
    assert DepthVisibility(frame).footprint_clear(obj)
    assert not DepthVisibility(
        replace(frame, depth_m=np.full((60, 100), np.nan, np.float32))
    ).footprint_clear(obj)
    assert not DepthVisibility(
        replace(frame, depth_m=np.full((60, 100), 10, np.float32))
    ).footprint_clear(obj)
    assert not DepthVisibility(frame).footprint_clear(
        footprint(corners=((-10, -1), (-8, -1), (-8, 1), (-10, 1)))
    )


def test_governor_resumes_ramp_from_measured_speed_not_zero():
    g = SpeedGovernor()
    for _ in range(100):
        g.limit(speed=5, dt=0.05)
    target, _ = g.limit(speed=4.8, dt=0.05, hold=True)
    assert target == 0
    target, _ = g.limit(speed=4.5, dt=0.05)
    assert target == pytest.approx(4.55)
    # From standstill the ramp is unchanged.
    g.reset()
    assert g.limit(speed=0, dt=0.05)[0] == pytest.approx(0.05)


def test_far_sighting_that_disappears_does_not_latch_a_stop():
    b = Behaviour()
    assert (
        b.update(time_s=0, speed=5, obstacle=True, must_stop=False, clear=False)
        == BehaviourState.APPROACH
    )
    assert (
        b.update(time_s=0.05, speed=5, obstacle=False, must_stop=False, clear=True)
        == BehaviourState.CRUISE
    )
    # Once a stop was required, a flickering detection cannot release it.
    b.update(time_s=0.1, speed=4, obstacle=True, must_stop=True, clear=False)
    assert (
        b.update(time_s=0.15, speed=3, obstacle=False, must_stop=False, clear=True)
        == BehaviourState.STOPPING
    )



def test_clearance_memory_is_invalidated_by_occlusion_or_invalid_depth():
    from obstacle_perception import visibility as vis

    route = RouteGeometry(((0, 0), (100, 0)))
    memory = vis.RouteClearanceMemory(15.0)

    class Scripted(vis.DepthVisibility):
        def __init__(self, status):
            self.status = status
            self.frame = SimpleFrame()

        def probe_status(self, point):
            return self.status

    sequence = [
        (vis.CLEAR, 8.0),
        (vis.OCCLUDED, 0.0),
        (vis.OUT_OF_VIEW, 0.0),
        (vis.INVALID, 0.0),
        (vis.OUT_OF_VIEW, 0.0),
        (vis.CLEAR, 8.0),
        (vis.OUT_OF_VIEW, 8.0),
    ]
    for step, (status, expected) in enumerate(sequence):
        horizon = Scripted(status).route_horizon(
            route, 0.0, 3.8, 1.2, 8.0, memory=memory, time_s=step * 0.05
        )
        assert horizon == expected, (step, status, horizon)


class SimpleFrame:
    ego = type("Ego", (), {"z_m": 0.0})()


def test_expired_track_cannot_reuse_confirmation_after_a_frame_gap():
    for identity in (1, None):
        f = TrackFilter(TrackFilterConfig(2, 3, 0.6))
        f.reset("e")
        for i in range(2):
            a = footprint(i, i * 0.05, identity=identity)
            rows = f.update((a,), a.metadata)
        assert rows[0].confirmed
        later = footprint(200, 10.0, identity=identity)
        rows = f.update((later,), later.metadata)
        assert len(rows) == 1 and not rows[0].confirmed
        again = footprint(201, 10.05, identity=identity)
        assert f.update((again,), again.metadata)[0].confirmed
