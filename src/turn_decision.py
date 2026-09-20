"""Junction detection and turn-direction decision logic for
demo_fused_lane_following.py.

Why this exists as a separate mechanism from ordinary lane-following: pure
lane-following only ever reacts to lane markings currently visible in the
forward-facing camera -- it has no way to DECIDE to turn, since a turn's
destination road usually isn't visible until the vehicle is already
partway through the turn. This module uses CARLA's own map waypoint graph
(the same `waypoint.next()` / `is_junction` mechanism already proven
working elsewhere in this project -- see demo_combined_perception.py's
measure_straightness/rank_straight_spawn_candidates and
collect_intersection_data.py's junction handling) to detect an upcoming
junction, classify its available exits as straight/right/left by heading
change, and pick one per the priority rule: straight always wins if
available; otherwise alternate right/left.

CARLA's yaw convention: increasing yaw = turning RIGHT (clockwise, viewed
from above) -- confirmed by carla_adapter/adapter.py's own
`yaw_rad = -math.radians(transform.rotation.yaw)` conversion (CARLA
clockwise-positive/left-handed -> controller counter-clockwise-
positive/right-handed).
"""



def _angle_diff_deg(a: float, b: float) -> float:
    """Smallest signed difference b - a, wrapped to (-180, 180].
    Identical to the helper already used by measure_straightness elsewhere
    in this file, duplicated here to keep this module import-independent.
    """
    d = (b - a) % 360.0
    if d > 180.0:
        d -= 360.0
    return d


def classify_candidate_direction(
    approach_yaw_deg: float, candidate_yaw_deg: float, straight_threshold_deg: float
) -> str:
    """Classify one candidate exit's direction relative to the approach
    heading: "straight", "right", or "left".
    """
    heading_change = _angle_diff_deg(approach_yaw_deg, candidate_yaw_deg)
    if abs(heading_change) <= straight_threshold_deg:
        return "straight"
    return "right" if heading_change > 0 else "left"


def find_junction_ahead(current_waypoint, lookahead_m: float, step_m: float = 2.0):
    """Walk forward in small steps from current_waypoint. Returns the
    waypoint immediately BEFORE a junction starts (a stable approach-
    heading reference point) if one is found within lookahead_m, else None.
    """
    current = current_waypoint
    distance_travelled = 0.0
    while distance_travelled < lookahead_m:
        next_options = current.next(step_m)
        if not next_options:
            return None
        next_wp = next_options[0]
        if next_wp.is_junction and not current.is_junction:
            return current
        current = next_wp
        distance_travelled += step_m
    return None


def get_junction_exit_candidates(pre_junction_waypoint, step_m: float = 2.0, max_steps: int = 40):
    """From a waypoint just before a junction, walk forward until next()
    fans out into multiple options (the junction's available paths).

    Returns (approach_yaw_deg, list_of_candidate_waypoints). The approach
    yaw is measured at pre_junction_waypoint itself -- just before entering
    the junction, so it reflects the real approach heading, not an
    already-turning one.
    """
    approach_yaw_deg = pre_junction_waypoint.transform.rotation.yaw
    current_options = [pre_junction_waypoint]
    for _ in range(max_steps):
        all_next = []
        for wp in current_options:
            all_next.extend(wp.next(step_m))
        if not all_next:
            return approach_yaw_deg, []
        if len(all_next) > 1:
            return approach_yaw_deg, all_next
        current_options = all_next
        if not current_options[0].is_junction:
            # Passed straight through without ever branching (e.g. a
            # gentle curve CARLA still flags as a junction, or a single-
            # lane connector) -- only one real option, return it as such.
            return approach_yaw_deg, current_options
    return approach_yaw_deg, current_options


def select_turn_direction(
    approach_yaw_deg: float,
    candidates: list,
    preferred_direction: str,
    straight_threshold_deg: float,
):
    """Pick which candidate exit to take, and which direction that
    represents.

    Priority: straight always wins if available (no alternation, the
    preferred_direction counter is untouched by the caller in this case).
    Otherwise, take preferred_direction if available; else the other of
    right/left; else (no real choice at all -- shouldn't normally happen)
    fall back to whichever candidate is closest to the approach heading.

    Returns (chosen_waypoint, direction_taken) or (None, None) if
    candidates is empty.
    """
    if not candidates:
        return None, None

    classified = [
        (classify_candidate_direction(approach_yaw_deg, wp.transform.rotation.yaw, straight_threshold_deg), wp)
        for wp in candidates
    ]

    straight_options = [wp for direction, wp in classified if direction == "straight"]
    if straight_options:
        return straight_options[0], "straight"

    preferred_options = [wp for direction, wp in classified if direction == preferred_direction]
    if preferred_options:
        return preferred_options[0], preferred_direction

    other_direction = "left" if preferred_direction == "right" else "right"
    other_options = [wp for direction, wp in classified if direction == other_direction]
    if other_options:
        return other_options[0], other_direction

    best = min(
        classified,
        key=lambda item: abs(_angle_diff_deg(approach_yaw_deg, item[1].transform.rotation.yaw)),
    )
    return best[1], best[0]


def build_turn_route(pre_junction_waypoint, chosen_exit_waypoint, post_junction_distance_m: float, step_m: float = 2.0):
    """Build a sequence of navigation_control Waypoints from just before
    the junction, through the chosen exit, extending post_junction_distance_m
    further along the new road -- fed to PathFollower.set_path() ONCE as a
    fixed route for the whole turn maneuver (unlike ordinary lane-following's
    per-cycle replan).

    Returns None if fewer than 2 distinct points survive (degenerate route).
    """
    carla_waypoints = [pre_junction_waypoint, chosen_exit_waypoint]
    current = chosen_exit_waypoint
    distance = 0.0
    while distance < post_junction_distance_m:
        next_options = current.next(step_m)
        if not next_options:
            break
        current = next_options[0]
        carla_waypoints.append(current)
        distance += step_m

    from navigation_control.types import Waypoint
    # Same CARLA -> controller world-frame convention as carla_adapter/
    # adapter.py: y_m = -location.y (CARLA left-handed -> controller
    # right-handed).
    route = [Waypoint(x_m=wp.transform.location.x, y_m=-wp.transform.location.y) for wp in carla_waypoints]

    deduped = [route[0]]
    for wp in route[1:]:
        if wp != deduped[-1]:
            deduped.append(wp)
    return deduped if len(deduped) >= 2 else None
