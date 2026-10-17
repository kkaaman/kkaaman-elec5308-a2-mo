"""CARLA ground truth is available only to the capture/evaluation channel."""

import math

from evaluation.ground_truth import front_edge_gap


def footprint_xy(actor):
    """Actor bounding box XY polygon, including its local centre and yaw."""
    import carla

    box = actor.bounding_box
    pose = actor.get_transform()
    yaw = math.radians(box.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    points = []
    for x, y in (
        (box.extent.x, -box.extent.y),
        (box.extent.x, box.extent.y),
        (-box.extent.x, box.extent.y),
        (-box.extent.x, -box.extent.y),
    ):
        p = carla.Location(
            x=box.location.x + c * x - s * y, y=box.location.y + s * x + c * y, z=box.location.z
        )
        pose.transform(p)
        points.append((p.x, -p.y))
    return tuple(points)


def truth_record(ego, obstacle, route, ego_state, ego_half_width=1.08173):
    if obstacle is None or not obstacle.is_alive:
        return dict(truth_gap_m=None, truth_in_corridor=False, truth_obstacle=None)
    points = footprint_xy(obstacle)
    ego_s = route.project(ego_state.x_m, ego_state.y_m).s_m
    interval = route.footprint_interval(
        points, half_width_m=ego_half_width + 0.15, minimum_s=max(0, ego_s - 2)
    )
    relevant = interval is not None and interval[1] >= ego_s
    front = footprint_xy(ego)[:2]
    return dict(
        truth_gap_m=front_edge_gap(front, points) if relevant else None,
        truth_in_corridor=relevant,
        truth_obstacle=dict(actor_id=obstacle.id, type_id=obstacle.type_id, corners_world_m=points),
    )
