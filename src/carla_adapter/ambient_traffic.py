"""Ambient traffic for a drive (4 Nov): CARLA autopilot cars and AI walkers spread over the
town, so the car meets ordinary traffic as well as its scripted actors. Evaluator side:
the car still sees them only through its own sensors. Off unless ``--traffic-vehicles`` /
``--traffic-walkers`` ask for it (the frozen runs are unchanged).

The spawning follows ``mapless_capture`` (Traffic Manager in synchronous mode, seeded;
walkers on the navigation mesh with an AI controller each). Nothing spawns within
``clear_radius_m`` of the ego, so the run does not start boxed in. ``stop`` stops the
walkers' controllers and takes the Traffic Manager out of synchronous mode before the
session restores the world's settings and destroys the actors.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficConfig:
    vehicles: int = 0
    walkers: int = 0
    seed: int = 5308
    cross_factor: float = 0.1  # share of walkers that cross roads (set_pedestrians_cross_factor)
    clear_radius_m: float = 35.0  # nothing spawns this near the ego
    tm_port: int = 8000
    speed_difference_pct: float = 20.0  # autopilot cars this much under each road's limit
    leading_gap_m: float = 3.0

    @property
    def enabled(self):
        return self.vehicles > 0 or self.walkers > 0


# Fire engines, ambulances, lorries, buses and vans: big and slow to read (a fire engine's
# 6.3 x 2.6 m footprint oncoming in the next lane was a "conflict" through every bend in
# traffic2 lap2_5308); the town is populated with cars.
LARGE = ("firetruck", "ambulance", "carlacola", "european_hgv", "fusorosa", "sprinter", "cybertruck", "t2", "truck", "bus", "van")


def _value(bp, name):
    """A blueprint attribute's value as text. CARLA returns an ``ActorAttribute``, whose
    ``str()`` is its description ("ActorAttribute(id=base_type,...)"), not its value:
    compared with "car" it matched no blueprint, and the first cars-only run stopped with
    "Cannot choose from an empty sequence" (traffic4, 6f68f51)."""
    attribute = bp.get_attribute(name)
    return str(attribute.as_str() if hasattr(attribute, "as_str") else attribute).strip()


def _by_name(bp):
    name = bp.id.lower()
    return not any(word in name for word in LARGE)


def _ordinary_car(bp):
    """A four-wheeled car: CARLA's ``base_type`` when the blueprint has one, else by name."""
    if not (bp.has_attribute("number_of_wheels") and _value(bp, "number_of_wheels") == "4"):
        return False
    if bp.has_attribute("base_type") and _value(bp, "base_type"):
        return _value(bp, "base_type").lower() == "car"
    return _by_name(bp)


class AmbientTraffic:
    def __init__(self, carla, owner, config):
        self.carla, self.owner, self.config = carla, owner, config
        self.tm = None
        self.vehicles, self.walkers, self.controllers = [], [], []

    def spawn(self, ego_location):
        """Spawn the cars and walkers; returns the report entry."""
        c, carla, world = self.config, self.carla, self.owner.world
        rng = random.Random(c.seed)
        library = world.get_blueprint_library()
        self.tm = self.owner.client.get_trafficmanager(c.tm_port)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(c.seed)
        self.tm.set_global_distance_to_leading_vehicle(c.leading_gap_m)
        self.tm.global_percentage_speed_difference(c.speed_difference_pct)

        def far(location):
            return math.hypot(location.x - ego_location.x, location.y - ego_location.y) > c.clear_radius_m

        spawn_points = [p for p in world.get_map().get_spawn_points() if far(p.location)]
        rng.shuffle(spawn_points)
        vehicles = list(library.filter("vehicle.*"))
        cars = [bp for bp in vehicles if _ordinary_car(bp)]
        if not cars:  # no base_type to go by: four-wheelers whose names are not large vehicles
            cars = [bp for bp in vehicles if bp.has_attribute("number_of_wheels") and _value(bp, "number_of_wheels") == "4" and _by_name(bp)]
        for spawn in spawn_points[: c.vehicles if cars else 0]:
            bp = rng.choice(cars)
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            actor = self.owner.try_spawn(bp, spawn)
            if actor is not None:
                actor.set_autopilot(True, c.tm_port)
                self.vehicles.append(actor)
        if c.walkers > 0:
            world.set_pedestrians_seed(c.seed)
            world.set_pedestrians_cross_factor(c.cross_factor)
            people = list(library.filter("walker.pedestrian.*"))
            controller_bp = library.find("controller.ai.walker")
            for _ in range(c.walkers * 3):  # navigation points can be taken or too near
                if len(self.walkers) >= c.walkers:
                    break
                location = world.get_random_location_from_navigation()
                if location is None or not far(location):
                    continue
                bp = rng.choice(people)
                if bp.has_attribute("is_invincible"):
                    bp.set_attribute("is_invincible", "false")
                walker = self.owner.try_spawn(bp, carla.Transform(location))
                if walker is None:
                    continue
                controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
                self.owner._actors.append(controller)
                self.walkers.append(walker)
                self.controllers.append(controller)
        self.owner.tick()
        for controller in self.controllers:
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(rng.uniform(0.9, 1.6))
        return dict(
            vehicles=len(self.vehicles), walkers=len(self.walkers), asked_vehicles=c.vehicles, asked_walkers=c.walkers,
            seed=c.seed, cross_factor=c.cross_factor, clear_radius_m=c.clear_radius_m,
        )

    def stop(self):
        """Stop the walkers and release the Traffic Manager (the session destroys the actors)."""
        errors = []
        for controller in self.controllers:
            try:
                controller.stop()
            except Exception as error:  # noqa: BLE001
                errors.append(str(error))
        if self.tm is not None:
            try:
                self.tm.set_synchronous_mode(False)
            except Exception as error:  # noqa: BLE001
                errors.append(str(error))
        return errors
