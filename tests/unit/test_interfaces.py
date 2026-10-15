"""BL-F-01 boundary cases without CARLA or ML dependencies."""
import unittest
from dataclasses import replace

import numpy as np

from autonomy.interfaces import (
    AutonomyStack,
    CameraCalibration,
    ControlCommand,
    ControlPacket,
    Diagnostics,
    EgoPose,
    FrameBundle,
    GraphEdge,
    GraphNode,
    LaneGraph,
    VehicleState,
    World,
)


def calibration():
    # Optical right -> ego right (-Y), optical down -> ego down (-Z).
    return CameraCalibration(100, 100, 2, 2, ((0,0,1,1.5),(-1,0,0,0),(0,-1,0,1.7),(0,0,0,1)))


def frame():
    return FrameBundle("episode-1", 7, 0.35, 100, EgoPose(VehicleState(0,0,0,0)), np.zeros((4,4,3), dtype=np.uint8), calibration())


class ContractTests(unittest.TestCase):
    def test_owned_readonly_rgb_snapshot(self):
        source = np.zeros((4,4,3), dtype=np.uint8)
        observation = replace(frame(), rgb=source)
        source[:] = 255
        self.assertEqual(int(observation.rgb.max()), 0)
        with self.assertRaises(ValueError):
            observation.rgb[0,0,0] = 1

    def test_depth_aligned_and_nan_allowed(self):
        depth = np.ones((4,4), dtype=np.float32)
        depth[0,0] = np.nan
        observation = replace(frame(), depth_m=depth, depth_frame_id=7)
        depth[:] = 2
        self.assertEqual(observation.depth_m[1,1], 1)
        self.assertTrue(np.isnan(observation.depth_m[0,0]))

    def test_depth_mismatched_frame_rejected(self):
        with self.assertRaises(ValueError):
            replace(frame(), depth_m=np.ones((4,4)), depth_frame_id=6)

    def test_depth_bad_values_and_dimensions_rejected(self):
        for depth in (np.zeros((4,4)), np.full((4,4), np.inf), np.ones((3,4))):
            with self.subTest(shape=depth.shape):
                with self.assertRaises(ValueError):
                    replace(frame(), depth_m=depth, depth_frame_id=7)

    def test_rgb_invalid_shape_dtype_empty(self):
        for rgb in (np.zeros((4,4)), np.zeros((4,4,3)), np.zeros((0,4,3), dtype=np.uint8)):
            with self.assertRaises(ValueError):
                replace(frame(), rgb=rgb)

    def test_invalid_frame_metadata(self):
        for change in ({"frame_id": True}, {"frame_id": -1}, {"sim_time_s": float('nan')}, {"episode_id": ""}):
            with self.assertRaises(ValueError):
                replace(frame(), **change)

    def test_calibration_rejects_reflection(self):
        with self.assertRaises(ValueError):
            CameraCalibration(100,100,2,2, ((1,0,0,0),(0,-1,0,0),(0,0,1,0),(0,0,0,1)))

    def test_graph_valid_and_unknown_endpoint_rejected(self):
        nodes = (GraphNode("a",0,0), GraphNode("b",2,0))
        edge = GraphEdge("a","b",((0,0),(2,0)),2)
        self.assertEqual(len(LaneGraph(nodes,(edge,)).edges), 1)
        with self.assertRaises(ValueError):
            LaneGraph(nodes,(replace(edge, end_id="missing"),))

    def test_graph_duplicates_and_bad_geometry_rejected(self):
        a, b = GraphNode("a",0,0), GraphNode("b",2,0)
        edge = GraphEdge("a","b",((0,0),(2,0)),2)
        for nodes, edges in (((a,a),()), ((a,b),(edge,edge)), ((a,b),(replace(edge, points_xy_m=((0,0),(3,0))),))):
            with self.assertRaises(ValueError):
                LaneGraph(nodes, edges)

    def test_command_expires_and_episode_isolation(self):
        packet = ControlPacket("one",7,0.35,0.45,ControlCommand(0,0,1),Diagnostics())
        self.assertTrue(packet.is_current("one",0.4))
        self.assertFalse(packet.is_current("two",0.4))
        self.assertFalse(packet.is_current("one",0.46))
        self.assertFalse(packet.is_current("one",0.34))
        with self.assertRaises(ValueError):
            replace(packet, valid_until_sim_time_s=0.3)

    def test_diagnostics_invalid_requires_reason(self):
        with self.assertRaises(ValueError):
            Diagnostics(observation_valid=False)
        with self.assertRaises(ValueError):
            Diagnostics((("detector", -1),))
        with self.assertRaises(ValueError):
            Diagnostics((("detector", 1),("detector", 2)))
        self.assertFalse(Diagnostics(observation_valid=False, reasons=("missing depth",)).observation_valid)

    def test_reuses_existing_controller_command(self):
        from navigation_control.types import ControlCommand as Original
        self.assertIs(ControlCommand, Original)
        with self.assertRaises(ValueError):
            ControlCommand(0, 1, 1)

    def test_fake_world_protocol_and_stack_episode_reset(self):
        class FakeWorld:
            def reset(self, *, seed):
                return frame(), LaneGraph((), ())
            def step(self, packet):
                if not packet.is_current("episode-1", 0.4):
                    raise ValueError("stale packet")
                return frame()
            def close(self):
                pass
        class FakeStack:
            def reset(self, *, episode_id, graph):
                self.episode = episode_id
            def step(self, observation):
                if observation.episode_id != self.episode:
                    raise ValueError("wrong episode")
                return ControlPacket(self.episode, observation.frame_id, observation.sim_time_s, observation.sim_time_s + 0.1, ControlCommand(0,0,1), Diagnostics())
            def close(self):
                pass
        world, stack = FakeWorld(), FakeStack()
        self.assertIsInstance(world, World)
        self.assertIsInstance(stack, AutonomyStack)
        observation, graph = world.reset(seed=1)
        stack.reset(episode_id=observation.episode_id, graph=graph)
        packet = stack.step(observation)
        self.assertEqual(world.step(packet).frame_id, 7)
        with self.assertRaises(ValueError):
            world.step(replace(packet, episode_id="old"))

    def test_contract_import_does_not_load_simulator(self):
        import os
        import subprocess
        import sys
        from pathlib import Path
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "src"))
        result = subprocess.run([sys.executable, "-c", "import sys; import autonomy.interfaces; assert 'carla' not in sys.modules"], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
