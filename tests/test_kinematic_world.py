import pytest

from autonomy.interfaces import ControlPacket
from navigation_control import ControlCommand
from sim_harness import KinematicWorld


def command(world, steering, throttle, brake, reason):
    return ControlPacket(
        world.evaluation().frame_id,
        ControlCommand(steering, throttle, brake),
        reason,
    )


def test_world_accelerates_and_returns_next_frame():
    world = KinematicWorld(dt_s=0.1)
    first = world.reset({"run_id": "straight"})

    world.apply(command(world, 0.0, 1.0, 0.0, "accelerate"))
    second = world.tick(timeout_s=1.0)

    assert first.frame_id == 0
    assert second.frame_id == 1
    assert second.ego.speed_mps > 0.0
    assert second.ego.x_m > 0.0


def test_world_turns_with_bicycle_dynamics():
    world = KinematicWorld(dt_s=0.1)
    world.reset({"run_id": "curve"})

    world.apply(command(world, 0.2, 1.0, 0.0, "turn left"))
    frame = world.tick(timeout_s=1.0)

    assert frame.ego.yaw_rad > 0.0
    assert frame.ego.y_m > 0.0


def test_world_exposes_synthetic_obstacles_only_at_detector_boundary():
    world = KinematicWorld()
    world.reset(
        {
            "run_id": "obstacles",
            "obstacles": [{"key": "lead", "class_name": "vehicle", "x_m": 12.0, "y_m": 0.0}],
        }
    )

    scene = world.detector_scene(0)

    assert len(scene) == 1
    assert scene[0].key == "lead"
    assert world.obstacle_distances_m()["lead"] == pytest.approx(12.0)
    assert world.evaluation().actors[0].kind == "vehicle"


def test_reset_clears_motion_and_stop_command_reduces_speed():
    world = KinematicWorld(dt_s=0.1)
    world.reset({"run_id": "stop-and-restart"})

    world.apply(command(world, 0.0, 1.0, 0.0, "accelerate"))
    moving = world.tick(timeout_s=1.0)

    world.apply(command(world, 0.0, 0.0, 1.0, "stop"))
    stopped = world.tick(timeout_s=1.0)

    restarted = world.reset({"run_id": "new-run"})

    assert stopped.ego.speed_mps < moving.ego.speed_mps
    assert restarted.frame_id == 0
    assert restarted.ego.speed_mps == 0.0