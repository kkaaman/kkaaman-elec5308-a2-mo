"""E2-03: ground removal, corridor crop, bounded clustering, footprints and sparse invalidity."""

import numpy as np
import pytest

from obstacle_perception.lidar import LidarConfig, LidarSweep
from obstacle_perception.lidar_obstacles import (
    CLASS_OBSTACLE,
    CLASS_RAISED,
    CLASS_ROAD,
    EgoBody,
    EgoRoutePose,
    LidarObstacleBatch,
    LidarObstaclePolicy,
    LidarTrackFilter,
    cluster_points,
    corridor_mask,
    extract_obstacles,
    lidar_metadata,
    project_points,
    remove_ground,
    seen_clear_distance,
    track_obstacles,
)
from obstacle_perception.lidar_synthetic import Scene, SceneBox, approach_sequence, synthetic_sweep
from obstacle_perception.tracking import TrackPolicy
from planning.corridor import RouteGeometry

STRAIGHT = RouteGeometry(((0, 0), (60, 0)))
FRONT = EgoBody().front_m


def scene_sweep(boxes=(), frame_id=1, sim_time_s=0.05, **kwargs):
    return synthetic_sweep(
        Scene(boxes=tuple(boxes)),
        episode_id="e",
        frame_id=frame_id,
        sim_time_s=sim_time_s,
        **kwargs,
    )


def test_ground_removal_separates_road_kerb_and_obstacles():
    sweep = scene_sweep([SceneBox(20.0, 0.0, 4.0, 2.0, 1.5, label="car")])
    segmentation = remove_ground(sweep)
    road, raised, obstacle = (
        segmentation.of(k) for k in (CLASS_ROAD, CLASS_RAISED, CLASS_OBSTACLE)
    )
    assert segmentation.accepted_bands >= 10 and np.nanmax(np.abs(segmentation.band_z_m)) < 0.05
    assert (np.abs(road[:, 1]) <= 3.5 + 0.1).all()  # road returns lie on the carriageway
    assert (
        np.abs(raised[:, 1]) > 3.4
    ).all()  # pavement (kerb height) is beside the road, not on it
    assert len(raised) > 100 and len(road) > 500
    assert (obstacle[:, 0] >= 17.9).all() and (np.abs(obstacle[:, 1]) <= 1.1).all()
    assert len(obstacle) >= 6
    body_points = sweep.points_xyzi[
        (np.abs(sweep.points_xyzi[:, 0]) < 1) & (np.abs(sweep.points_xyzi[:, 1]) < 1)
    ]
    assert len(body_points) == 0 or segmentation.dropped_body >= 0


def test_sloped_road_is_still_ground_not_an_obstacle():
    sweep = synthetic_sweep(Scene(road_slope=0.03), episode_id="e", frame_id=1, sim_time_s=0.05)
    segmentation = remove_ground(sweep)
    assert len(segmentation.of(CLASS_OBSTACLE)) == 0
    assert segmentation.accepted_bands >= 8
    far = segmentation.points_xyz[:, 0] > 20
    assert (
        segmentation.classes[far & (np.abs(segmentation.points_xyz[:, 1]) < 3)] == CLASS_ROAD
    ).all()


def test_project_points_matches_route_geometry_and_corridor_mask_follows_bend():
    route = RouteGeometry(((0, 0), (10, 0), (10, 20)))
    xy = np.array([[2, -1], [9, 8], [10.5, 30], [30, 0]], float)
    s, lateral, distance = project_points(route, xy)
    for i, (x, y) in enumerate(xy):
        reference = route.project(x, y)
        assert s[i] == pytest.approx(reference.s_m)
        assert lateral[i] == pytest.approx(reference.lateral_m)
        assert distance[i] == pytest.approx(reference.distance_m)
    inside = corridor_mask(route, xy, half_width_m=1.5, ahead_m=40)
    assert inside.tolist() == [True, True, False, False]
    pose = EgoRoutePose(5.0, 0.0, 0.0)
    assert corridor_mask(
        route, [[0, 0]], ego_pose=pose, ego_s_m=5.0, half_width_m=1.5, ahead_m=40
    ).tolist() == [False]
    assert corridor_mask(
        route, [[1, 0]], ego_pose=pose, ego_s_m=5.0, half_width_m=1.5, ahead_m=40
    ).tolist() == [True]


def test_clustering_is_bounded_and_drops_noise():
    rng = np.random.default_rng(0)
    a = rng.normal((5, 0), 0.1, (20, 2))
    b = rng.normal((5, 6), 0.1, (20, 2))
    noise = np.array([[30, 30], [40, -40]], float)
    clusters = cluster_points(np.vstack((a, b, noise)), cell_m=0.5, min_points=6)
    assert sorted(len(c) for c in clusters) == [20, 20]
    assert cluster_points(np.zeros((0, 2)), cell_m=0.5, min_points=1) == []
    with pytest.raises(ValueError):
        cluster_points(a, cell_m=0, min_points=1)


def test_extraction_separates_in_corridor_cluster_from_roadside_objects():
    sweep = scene_sweep(
        [
            SceneBox(20.0, 0.0, 4.0, 2.0, 1.5, label="car"),
            SceneBox(14.0, 5.2, 4.4, 1.9, 1.5, base_z_m=0.15, label="parked"),
            SceneBox(9.0, -4.0, 0.3, 0.3, 3.0, base_z_m=0.15, label="pole"),
        ]
    )
    batch = extract_obstacles(sweep, STRAIGHT)
    assert batch.valid and batch.invalid_reason is None
    in_corridor = [c for c in batch.clusters if c.in_corridor]
    roadside = [c for c in batch.clusters if not c.in_corridor]
    assert len(in_corridor) == 1 and len(roadside) == 2
    car = in_corridor[0]
    assert car.corridor_gap_m == pytest.approx(18.0 - FRONT, abs=0.05)
    assert car.nearest_surface_m == pytest.approx(18.0, abs=0.05)
    assert car.nearest_point_ego_m[0] == pytest.approx(18.0, abs=0.05)
    assert car.footprint_ego_m[0][0] == pytest.approx(18.0, abs=0.05)
    assert abs(car.center_ego_m[1]) < 0.3 and car.extent_m[1] == pytest.approx(2.0, abs=0.3)
    assert batch.nearest_gap_m == car.corridor_gap_m and batch.nearest_cluster is car
    assert {round(c.center_ego_m[1]) for c in roadside} == {5, -4}
    assert all(c.corridor_gap_m is None for c in roadside)
    assert batch.visible_distance_m <= batch.nearest_gap_m
    assert not batch.usable_for_clearance
    obstacle = car.to_obstacle(batch.metadata, 0)
    assert obstacle.identity.source_id == "lidar" and obstacle.class_name == "lidar_cluster"
    assert 0 < obstacle.confidence <= 1
    assert batch.to_dict()["clusters"][0]["corridor_gap_m"] == car.corridor_gap_m


def test_extraction_follows_route_frame_and_bend():
    route = RouteGeometry(((0, 0), (10, 0), (10, 30)))
    # Ego sits 5 m along the straight leg; a car stands on the second leg after the bend.
    pose = EgoRoutePose(5.0, 0.0, 0.0)
    sweep = scene_sweep([SceneBox(5.0, 8.0, 2.0, 4.0, 1.5, label="car_after_bend")])
    batch = extract_obstacles(sweep, route, ego_pose=pose, ego_s_m=5.0)
    car = batch.nearest_cluster
    assert car is not None
    # Nearest corridor point at ego (5, 6): route (10, 6) -> arc 16; gap = 16 - 5 - front.
    assert car.corridor_gap_m == pytest.approx(16.0 - 5.0 - FRONT, abs=0.3)


def test_sparse_or_empty_sweeps_are_invalid_not_clear():
    boxes = [SceneBox(20.0, 0.0, 4.0, 2.0, 1.5)]
    sparse = scene_sweep(boxes, dropout=0.97)
    batch = extract_obstacles(sparse, STRAIGHT)
    assert not batch.valid and batch.invalid_reason == "sparse_sweep_points"
    assert batch.clusters == () and batch.nearest_gap_m is None and batch.seen_clear_m is None
    assert batch.visible_distance_m is None
    empty = LidarSweep(
        "e", 1, 0.05, 0.0, np.zeros((0, 4), np.float32), tuple(map(tuple, np.eye(4))), LidarConfig()
    )
    assert extract_obstacles(empty, STRAIGHT).invalid_reason == "sparse_sweep_points"
    dense_but_thin = extract_obstacles(
        scene_sweep(boxes, dropout=0.6),
        STRAIGHT,
        policy=LidarObstaclePolicy(min_points=100, min_density_fraction=0.5),
    )
    assert dense_but_thin.invalid_reason == "sparse_sweep_density"
    with pytest.raises(ValueError):
        LidarObstacleBatch(lidar_metadata(sparse), False, None, (), 0, 0.0, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        LidarObstacleBatch(
            lidar_metadata(sparse),
            True,
            None,
            (),
            0,
            0.0,
            0,
            0,
            0,
            0,
            1.0,
            usable_for_clearance=True,
        )


def test_road_not_established_when_only_the_vehicle_band_is_seen():
    high = EgoBody(front_m=3.8)
    policy = LidarObstaclePolicy(min_points=10, min_density_fraction=0.001)
    points = np.column_stack(
        (np.full(50, 30.0), np.linspace(-1, 1, 50), np.full(50, 1.0), np.full(50, 0.5))
    )
    sweep = LidarSweep(
        "e", 1, 0.05, 0.0, points.astype(np.float32), tuple(map(tuple, np.eye(4))), LidarConfig()
    )
    batch = extract_obstacles(sweep, STRAIGHT, policy=policy, body=high)
    assert not batch.valid and batch.invalid_reason == "road_not_established"


def test_seen_clear_requires_lateral_coverage_and_scales_gap_with_range():
    policy = LidarObstaclePolicy()
    body = EgoBody()
    # Road returns in all three strips every 0.5 m from the bumper to 20 m.
    s = np.repeat(np.arange(body.front_m + 0.25, body.front_m + 20, 0.5), 3)
    y = np.tile([-1.0, 0.0, 1.0], len(s) // 3)
    full = np.column_stack((np.repeat(s, 2), np.repeat(y, 2)))
    assert seen_clear_distance(full, STRAIGHT, policy=policy, body=body) == pytest.approx(
        20.0, abs=0.5
    )
    # Only the outer strips: an unobserved middle is not clearance.
    sides = full[np.abs(full[:, 1]) > 0.5]
    assert seen_clear_distance(sides, STRAIGHT, policy=policy, body=body) == 0.0
    # A ring gap that grows with range is tolerated near the limit, not beyond it.
    ringed = full[(full[:, 0] < body.front_m + 10) | (full[:, 0] > body.front_m + 13)]
    assert seen_clear_distance(ringed, STRAIGHT, policy=policy, body=body) == pytest.approx(
        20.0, abs=0.5
    )
    gapped = full[(full[:, 0] < body.front_m + 5) | (full[:, 0] > body.front_m + 12)]
    assert seen_clear_distance(gapped, STRAIGHT, policy=policy, body=body) == pytest.approx(
        5.0, abs=0.5
    )
    assert seen_clear_distance(np.zeros((0, 2)), STRAIGHT, policy=policy, body=body) == 0.0


def test_raised_cluster_in_corridor_counts_once_with_its_obstacle():
    sweep = scene_sweep([SceneBox(20.0, 0.0, 4.0, 2.0, 1.5)])
    batch = extract_obstacles(sweep, STRAIGHT)
    assert sum(c.in_corridor for c in batch.clusters) == 1
    # A lone kerb-height object (0.25 m) in the lane is still reported as raised.
    low = extract_obstacles(scene_sweep([SceneBox(12.0, 0.0, 1.0, 2.5, 0.25)]), STRAIGHT)
    raised = [c for c in low.clusters if c.in_corridor]
    assert len(raised) == 1 and raised[0].raised
    assert raised[0].to_obstacle(low.metadata).class_name == "lidar_raised"


def test_cluster_cap_marks_truncation():
    boxes = [SceneBox(8.0 + 2 * k, 6.0, 0.4, 0.4, 2.0, base_z_m=0.15) for k in range(12)]
    batch = extract_obstacles(
        scene_sweep(boxes), STRAIGHT, policy=LidarObstaclePolicy(max_clusters=4)
    )
    assert batch.truncated and len(batch.clusters) == 4


def test_tracking_confirms_namespaces_ids_and_treats_invalid_sweeps_as_misses():
    f = LidarTrackFilter(policy=TrackPolicy(hits=2, window=3, expiry_s=0.5))
    f.reset(episode_id="e")
    rows = approach_sequence(frames=4)
    batches = [
        extract_obstacles(
            synthetic_sweep(scene, episode_id="e", frame_id=fid, sim_time_s=t), STRAIGHT
        )
        for fid, t, scene, _ in rows
    ]
    first = f.update(batches[0])
    assert (
        all(not t.confirmed for t in first.tracks)
        and first.unknown_tracks
        and first.nearest_gap_m is None
    )
    second = f.update(batches[1])
    target = [t for t in second.tracks if t.gap_valid]
    assert len(target) == 1 and target[0].confirmed and target[0].identity.source_id == "lidar"
    assert target[0].identity.model_version == "lidar-obstacles-v1"
    assert target[0].gap_m == pytest.approx(rows[1][3], abs=0.05)
    assert not second.unknown_tracks and not second.usable_for_clearance
    assert second.visible_distance_m <= second.nearest_gap_m
    obstacles = track_obstacles(second)
    assert {o.identity for o in obstacles} == {t.identity for t in second.tracks if t.confirmed}
    # Invalid sweep: tracks miss, no gap, batch invalid, nothing cleared.
    sparse = extract_obstacles(
        synthetic_sweep(rows[2][2], episode_id="e", frame_id=15, sim_time_s=0.75, dropout=0.97),
        STRAIGHT,
    )
    third = f.update(sparse)
    assert not third.valid and third.nearest_gap_m is None and third.visible_distance_m is None
    assert all(t.invalid_reason == "invalid_sweep" for t in third.tracks)
    assert third.unknown_tracks  # the corridor track is no longer validly observed
    fourth = f.update(batches[3])
    assert fourth.nearest_gap_m == pytest.approx(rows[3][3], abs=0.05)
    # Expiry after a long gap removes the history; new IDs continue to increase.
    late = extract_obstacles(
        synthetic_sweep(rows[3][2], episode_id="e", frame_id=99, sim_time_s=5.0), STRAIGHT
    )
    fifth = f.update(late)
    assert len(fifth.expired) == len(fourth.tracks)
    assert all(not t.confirmed for t in fifth.tracks)
    assert min(t.identity.track_id for t in fifth.tracks) > max(t.track_id for t in fifth.expired)


def test_tracking_rejects_wrong_episode_and_non_monotonic_sweeps():
    f = LidarTrackFilter()
    f.reset(episode_id="e")
    batch = extract_obstacles(scene_sweep([], frame_id=5, sim_time_s=0.25), STRAIGHT)
    f.update(batch)
    with pytest.raises(ValueError):
        f.update(batch)
    other = extract_obstacles(
        synthetic_sweep(Scene(), episode_id="other", frame_id=6, sim_time_s=0.3), STRAIGHT
    )
    with pytest.raises(ValueError):
        f.update(other)
    with pytest.raises(TypeError):
        f.update(object())
    with pytest.raises(ValueError):
        LidarTrackFilter(gate_m=0)
    with pytest.raises(ValueError):
        LidarObstaclePolicy(ground_max_m=0.5)
    with pytest.raises(ValueError):
        EgoBody(rear_m=1.0)
