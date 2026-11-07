"""Collision and lane-invasion event records for S0-S5 (BL-B-07). Simulator-free.

Records use the same schema as the live S0 runner (``carla_adapter.s0_runner``),
so ``evaluation.s0_metrics`` counts them with the same rules:
- **collision:** one record per sensor callback. All callbacks of one contact share
  a ``contact_id``. Callbacks against the same other actor belong to one contact
  while consecutive callbacks are at most ``window_s`` (0.5 s,
  ``configs/acceptance.json`` ``collision_dedup_s``) of simulation time apart. A
  later touch starts a new contact. A callback with no actor id is its own contact.
- **lane invasion:** one record per callback, with a unique ``event_id``. It is
  excluded from the unplanned count only if it falls inside a declared legal
  manoeuvre: an entry of the route's ``legal_lane_invasion_exclusions`` covering
  its route arc length, for every crossed marking type. Junctions are never
  exempted wholesale.

Raw records are always kept, for failed as well as successful runs; counts are
derived from them, never stored instead of them.
"""

import math

COLLISION_DEDUP_S = 0.5


class CollisionContacts:
    """Assign ``contact_id`` to collision callbacks with the documented rule."""

    def __init__(self, window_s=COLLISION_DEDUP_S):
        if isinstance(window_s, bool) or not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be positive")
        self.window_s = float(window_s)
        self._open = {}
        self._anonymous = 0

    def contact_id(self, other_actor_id, frame, sim_time_s):
        if other_actor_id is None:
            self._anonymous += 1
            return f"unknown:{frame}:{self._anonymous}"
        key = str(other_actor_id)
        previous = self._open.get(key)
        start = (
            frame if previous is None or sim_time_s - previous[1] > self.window_s else previous[0]
        )
        self._open[key] = (start, sim_time_s)
        return f"{key}:{start}"


def collision_record(contacts, *, frame, sim_time_s, other_actor_id, other_actor_type, impulse):
    return {
        "kind": "collision",
        "frame": frame,
        "simulation_time_s": sim_time_s,
        "contact_id": contacts.contact_id(other_actor_id, frame, sim_time_s),
        "other_actor_id": None if other_actor_id is None else str(other_actor_id),
        "other_actor_type": other_actor_type,
        "normal_impulse": dict(zip("xyz", impulse)),
    }


def lane_invasion_record(*, sequence, frame, sim_time_s, markings, route_s_m, exclusions):
    """One lane-invasion callback, annotated against the declared legal manoeuvres."""
    markings = [str(m) for m in markings]
    match = legal_exclusion(route_s_m, markings, exclusions)
    record = {
        "kind": "lane_invasion",
        "frame": frame,
        "simulation_time_s": sim_time_s,
        "event_id": f"{frame}:{sequence}",
        "crossed_lane_markings": markings,
        "route_s_m": route_s_m,
        "predeclared_legal": match is not None,
    }
    if match is not None:
        record["legal_manoeuvre_id"] = match
    return record


def validate_exclusions(exclusions):
    """Check the route's ``legal_lane_invasion_exclusions`` entries; returns a tuple."""
    result = []
    for entry in exclusions or ():
        if not isinstance(entry, dict) or set(entry) != {
            "id",
            "from_s_m",
            "to_s_m",
            "markings",
            "reason",
        }:
            raise ValueError(
                "Each legal exclusion needs exactly id, from_s_m, to_s_m, markings and reason"
            )
        low, high = entry["from_s_m"], entry["to_s_m"]
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (low, high)):
            raise ValueError("Legal exclusion bounds must be finite")
        if high <= low or not entry["markings"] or not str(entry["reason"]).strip():
            raise ValueError("Legal exclusion needs an interval, marking types and a reason")
        if not str(entry["id"]).strip():
            raise ValueError("Legal exclusion needs an id")
        result.append(dict(entry, markings=tuple(str(m) for m in entry["markings"])))
    return tuple(result)


def legal_exclusion(route_s_m, markings, exclusions):
    """Id of the declared manoeuvre covering this crossing, or None."""
    if route_s_m is None or not markings:
        return None
    for entry in exclusions:
        if entry["from_s_m"] <= route_s_m <= entry["to_s_m"] and all(
            m in entry["markings"] for m in markings
        ):
            return entry["id"]
    return None


def count_contacts(records):
    return len({r["contact_id"] for r in records if r.get("kind") == "collision"})


def count_unplanned_invasions(records):
    return len(
        {
            r["event_id"]
            for r in records
            if r.get("kind") == "lane_invasion" and not r.get("predeclared_legal")
        }
    )
