"""BL-A-01 orchestration and forward-controller audit checks."""
import json
import math
import unittest
from pathlib import Path

from navigation_control import PathFollower
from navigation_control.types import ControllerConfig, ControlStatus, VehicleState, Waypoint


def follower():
    controller = PathFollower(ControllerConfig(wheelbase_m=2.8, max_steering_angle_rad=0.5))
    controller.set_path([Waypoint(0,0), Waypoint(50,0)])
    return controller


class ForwardMissionTests(unittest.TestCase):
    def test_hold_resume_preserves_route_and_progress_without_set_path(self):
        controller = follower()
        route = controller._path
        cruise = controller.update(VehicleState(10,0,0,2), 3, 0.05)
        hold = controller.update(VehicleState(12,0,0,1), 0, 0.05)
        self.assertEqual(hold.command.throttle, 0)
        self.assertGreater(hold.command.brake, 0)
        resume = controller.update(VehicleState(12,0,0,0), 3, 0.05)
        self.assertIs(controller._path, route)
        self.assertGreaterEqual(hold.diagnostics.progress_m, cruise.diagnostics.progress_m)
        self.assertEqual(resume.diagnostics.progress_m, hold.diagnostics.progress_m)
        self.assertGreater(resume.command.throttle, 0)
        self.assertNotEqual(resume.status, ControlStatus.FINISHED)

    def test_left_right_correction_uses_positive_left_angle(self):
        for lateral, expected_sign in ((-1,1),(1,-1)):
            controller = follower()
            result = controller.update(VehicleState(10,lateral,0,2), 3, 0.05)
            self.assertGreater(result.command.steering_angle_rad * expected_sign, 0)

    def test_endpoint_speed_ceiling_and_latched_goal(self):
        controller = follower()
        result = controller.update(VehicleState(49,0,0,1), 3, 0.05)
        self.assertLess(result.diagnostics.internal_target_speed_mps, 3)
        final = controller.update(VehicleState(50,0,0,0), 3, 0.05)
        self.assertEqual(final.status, ControlStatus.FINISHED)
        later = controller.update(VehicleState(40,0,0,0), 3, 0.05)
        self.assertEqual(later.status, ControlStatus.FINISHED)
        self.assertEqual(later.command.throttle, 0)
        self.assertGreater(later.command.brake, 0)
        controller.reset()
        restart = controller.update(VehicleState(0,0,0,0), 3, 0.05)
        self.assertEqual(restart.diagnostics.progress_m, 0)
        self.assertNotEqual(restart.status, ControlStatus.FINISHED)

    def test_negative_speed_rejected(self):
        with self.assertRaises(ValueError):
            VehicleState(0,0,0,-1)
        with self.assertRaises(ValueError):
            follower().update(VehicleState(0,0,0,0), -1, 0.05)

    def test_invalid_input_does_not_advance_progress(self):
        controller = follower()
        controller.update(VehicleState(10,0,0,2), 3, 0.05)
        with self.assertRaises(ValueError):
            controller.update(VehicleState(30,0,0,2), 3, math.nan)
        result = controller.update(VehicleState(9,0,0,2), 3, 0.05)
        self.assertEqual(result.diagnostics.progress_m, 10)

    def test_spec_is_frozen_forward_open_and_truth_separated(self):
        path = Path(__file__).resolve().parents[2] / 'configs/mission_baseline.json'
        spec = json.loads(path.read_text())
        self.assertEqual(spec['direction'], 'forward')
        self.assertEqual(spec['route_kind'], 'open_non_self_crossing')
        self.assertEqual(spec['tentative_manoeuvres'][-1], 'stop_at_goal')
        self.assertFalse(spec['reverse_allowed'])
        self.assertFalse(spec['obstacle_truth_for_control'])
        self.assertEqual(spec['route_selection_status'], 'FROZEN_PENDING_S0_ACCEPTANCE')
        self.assertEqual(spec['ego_truth_for_navigation'], 'ALLOWED_BY_USER_COURSE_SCOPE_CONFIRMATION')
        self.assertIn('route_file', spec)
        self.assertEqual(spec['set_path_policy'], 'once_per_episode_before_first_update')
