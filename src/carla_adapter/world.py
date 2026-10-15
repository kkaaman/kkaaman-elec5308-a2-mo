"""Sensor-free synchronous world ownership and cleanup for BL-B-01."""

import math
import threading
from dataclasses import dataclass

_OWNERS = set()
_OWNER_LOCK = threading.Lock()


@dataclass(frozen=True)
class WorldTiming:
    fixed_delta_seconds: float = 0.05
    max_substep_delta_time: float = 0.01
    max_substeps: int = 10
    tick_timeout_seconds: float = 10.0

    def __post_init__(self):
        for value in (
            self.fixed_delta_seconds,
            self.max_substep_delta_time,
            self.tick_timeout_seconds,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("World timing values must be finite and positive")
        if isinstance(self.max_substeps, bool) or not isinstance(
            self.max_substeps, int
        ):
            raise ValueError("max_substeps must be an integer")
        if not 1 <= self.max_substeps <= 10:
            raise ValueError("max_substeps must lie between 1 and 10")
        if self.max_substep_delta_time > 0.01:
            raise ValueError("Physics substeps must be at most 0.01 seconds")
        if (
            self.fixed_delta_seconds
            > self.max_substep_delta_time * self.max_substeps + 1e-12
        ):
            raise ValueError("Fixed step exceeds physics substep budget")


class SynchronousWorldSession:
    """Own world ticks and only actors spawned through this session.

    One session per endpoint in this process; refuses an already synchronous
    world. Other clients must not tick or change settings during the session.
    No sensors or Traffic Manager are enabled here. Use as a context manager.
    """

    def __init__(
        self,
        client,
        timing=None,
        ownership_key="127.0.0.1:2000",
        brake_control_factory=None,
    ):
        self.client = client
        self.timing = timing or WorldTiming()
        self.ownership_key = ownership_key
        self._brake_factory = brake_control_factory
        self.world = None
        self._original_settings = None
        self._actors = []
        self._thread_id = None
        self._active = False
        self._used = False
        self._claimed = False
        self._last_frame = None
        self.stop_errors = []

    def _brake(self):
        if self._brake_factory is not None:
            return self._brake_factory()
        import carla

        return carla.VehicleControl(brake=1.0, throttle=0.0, reverse=False)

    def open(self):
        if self._used:
            raise RuntimeError("Sessions are single-use; construct a new session")
        self._used = True
        with _OWNER_LOCK:
            if self.ownership_key in _OWNERS:
                raise RuntimeError("Another session owns this endpoint")
            _OWNERS.add(self.ownership_key)
            self._claimed = True
        try:
            self.world = self.client.get_world()
            self._original_settings = self.world.get_settings()
            if self._original_settings.synchronous_mode:
                self._original_settings = None
                raise RuntimeError(
                    "World already synchronous; tick ownership is ambiguous"
                )
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.timing.fixed_delta_seconds
            settings.substepping = True
            settings.max_substep_delta_time = self.timing.max_substep_delta_time
            settings.max_substeps = self.timing.max_substeps
            self.world.apply_settings(settings)
            self._thread_id = threading.get_ident()
            self._active = True
            return self
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"Startup cleanup failed: {cleanup_error}")
            raise

    def _require_owner(self):
        if not self._active:
            raise RuntimeError("Session is not active")
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("Only the opening thread may advance or spawn")

    def try_spawn(self, blueprint, transform):
        self._require_owner()
        if blueprint.id.startswith("sensor."):
            raise ValueError(
                "Sensors are disabled until their capture configuration exists"
            )
        actor = self.world.try_spawn_actor(blueprint, transform)
        if actor is not None:
            self._actors.append(actor)
        return actor

    def emergency_stop(self):
        """Attempt full braking on every owned live vehicle; retain failures."""
        errors = []
        for actor in self._actors:
            try:
                if actor.is_alive and actor.type_id.startswith("vehicle."):
                    actor.apply_control(self._brake())
            except Exception as error:
                errors.append(f"actor {actor.id}: {error}")
        self.stop_errors.extend(errors)
        return tuple(errors)

    def tick(self):
        self._require_owner()
        try:
            frame = self.world.tick(self.timing.tick_timeout_seconds)
            if self._last_frame is not None and frame != self._last_frame + 1:
                raise RuntimeError("Nonconsecutive frame: possible external tick owner")
            self._last_frame = frame
            return frame
        except Exception:
            self.emergency_stop()
            raise

    def close(self):
        """Stop, destroy owned actors, restore settings; attempt every cleanup step."""
        errors = list(self.emergency_stop())
        for actor in reversed(self._actors):
            try:
                if actor.is_alive and not actor.destroy():
                    errors.append(f"Destroy failed for actor {actor.id}")
            except Exception as error:
                errors.append(f"Destroy actor {actor.id}: {error}")
        self._actors.clear()
        try:
            if self._original_settings is not None:
                self.world.apply_settings(self._original_settings)
        except Exception as error:
            errors.append(f"Restore settings: {error}")
        finally:
            self._original_settings = None
            self._active = False
            if self._claimed:
                with _OWNER_LOCK:
                    _OWNERS.discard(self.ownership_key)
                self._claimed = False
        if errors:
            raise RuntimeError("Cleanup incomplete: " + "; ".join(errors))

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.close()
        except Exception as error:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(str(error))
        return False


def _settings_record(settings):
    """Record every exposed scalar setting, including unrelated settings."""
    result = {}
    for name in dir(settings):
        if not name.startswith("_"):
            value = getattr(settings, name)
            if value is None or isinstance(value, (bool, int, float, str)):
                result[name] = value
    return result


def main():
    """Live repeated start/stop plus injected timeout; no sensors or driving."""
    import argparse
    import json
    from pathlib import Path

    import carla

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Evidence already exists; choose a new output name")
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(15)
    world = client.get_world()
    initial = _settings_record(world.get_settings())
    initial_ids = {actor.id for actor in world.get_actors()}
    results = []

    class TimeoutOnce:
        def __init__(self, real_world):
            self.real_world = real_world
            self.pending = True

        def __getattr__(self, name):
            return getattr(self.real_world, name)

        def tick(self, timeout):
            if self.pending:
                self.pending = False
                raise RuntimeError("Injected tick timeout")
            return self.real_world.tick(timeout)

    for cycle in range(4):
        with SynchronousWorldSession(client) as owner:
            blueprint = owner.world.get_blueprint_library().find("vehicle.tesla.model3")
            vehicle = None
            for spawn in owner.world.get_map().get_spawn_points():
                vehicle = owner.try_spawn(blueprint, spawn)
                if vehicle is not None:
                    break
            if vehicle is None:
                raise RuntimeError("No free spawn point")
            vehicle.apply_control(carla.VehicleControl(brake=1))
            frames = [owner.tick() for _ in range(10)]
            delta = owner.world.get_snapshot().timestamp.delta_seconds
            if not math.isclose(delta, owner.timing.fixed_delta_seconds, abs_tol=1e-6):
                raise RuntimeError("Unexpected simulation timestep")
            timeout_stop = "NOT_APPLICABLE"
            if cycle == 3:
                # Inject a timeout while the real server stays reachable. This
                # verifies braking dispatch, not disconnected-server delivery.
                vehicle.apply_control(carla.VehicleControl(brake=0, throttle=0))
                owner.tick()
                owner.world = TimeoutOnce(owner.world)
                try:
                    owner.tick()
                except RuntimeError as error:
                    if "Injected tick timeout" not in str(error):
                        raise
                else:
                    raise RuntimeError("Timeout injection did not execute")
                owner.tick()
                control = vehicle.get_control()
                if control.brake < 0.99 or control.throttle != 0 or owner.stop_errors:
                    raise RuntimeError("Timeout did not apply full braking")
                timeout_stop = "PASS_INJECTED_TIMEOUT_REACHABLE_SERVER"
            actor_id = vehicle.id
        restored = _settings_record(world.get_settings()) == initial
        final_ids = {actor.id for actor in world.get_actors()}
        cleaned = actor_id not in final_ids and final_ids == initial_ids
        results.append(
            {
                "cycle": cycle + 1,
                "frames": frames,
                "delta_seconds": delta,
                "settings_restored": restored,
                "no_actor_leaks": cleaned,
                "timeout_stop": timeout_stop,
            }
        )
        if not restored or not cleaned:
            raise RuntimeError(
                "World settings or actor inventory changed after cleanup"
            )
    report = {
        "server": client.get_server_version(),
        "map": world.get_map().name,
        "initial_settings": initial,
        "cycles": results,
        "status": "PASS",
        "scope": "sensor-free lifecycle; timeout is injected with reachable server",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
