"""Lifecycle and fault injection without CARLA or a network."""

import copy
from types import SimpleNamespace

import pytest

from carla_adapter.world import SynchronousWorldSession, WorldTiming


class FakeWorld:
    def __init__(self):
        self.settings = SimpleNamespace(
            synchronous_mode=False,
            fixed_delta_seconds=None,
            substepping=False,
            max_substep_delta_time=0.02,
            max_substeps=3,
            no_rendering_mode=False,
        )
        self.actor = SimpleNamespace(
            id=7, type_id="vehicle.tesla.model3", is_alive=True
        )
        self.controls = []
        self.actor.apply_control = self.controls.append
        self.actor.destroy = self.destroy
        self.frame = 10
        self.fail_tick = False
        self.fail_apply_once = False
        self.destroyed = 0

    def get_settings(self):
        return copy.deepcopy(self.settings)

    def apply_settings(self, settings):
        self.settings = copy.deepcopy(settings)
        if self.fail_apply_once:
            self.fail_apply_once = False
            raise RuntimeError("apply timeout after partial change")

    def try_spawn_actor(self, blueprint, transform):
        return self.actor

    def destroy(self):
        self.destroyed += 1
        self.actor.is_alive = False
        return True

    def tick(self, timeout):
        if self.fail_tick:
            raise RuntimeError("tick timeout")
        self.frame += 1
        return self.frame


def session(world, key="fake-world"):
    return SynchronousWorldSession(
        SimpleNamespace(get_world=lambda: world),
        ownership_key=key,
        brake_control_factory=lambda: SimpleNamespace(brake=1, throttle=0),
    )


def spawn(owner):
    return owner.try_spawn(SimpleNamespace(id="vehicle.tesla.model3"), object())


def test_restore_actor_cleanup_and_idempotent_close():
    world = FakeWorld()
    before = vars(world.get_settings())
    with session(world) as owner:
        spawn(owner)
        assert world.settings.synchronous_mode
        assert world.settings.fixed_delta_seconds == 0.05
        assert owner.tick() == 11
        assert owner.tick() == 12
    assert vars(world.get_settings()) == before
    assert world.destroyed == 1
    assert world.controls[-1].brake == 1
    owner.close()
    assert world.destroyed == 1


def test_timeout_brakes_then_context_restores():
    world = FakeWorld()
    with pytest.raises(RuntimeError, match="tick timeout"):
        with session(world) as owner:
            spawn(owner)
            world.fail_tick = True
            owner.tick()
    assert len(world.controls) >= 2
    assert all(c.brake == 1 and c.throttle == 0 for c in world.controls)
    assert not world.settings.synchronous_mode
    assert not world.actor.is_alive


def test_duplicate_owner_and_existing_synchronous_world_rejected():
    world = FakeWorld()
    with session(world):
        with pytest.raises(RuntimeError, match="Another session"):
            session(world).open()
    world.settings.synchronous_mode = True
    with pytest.raises(RuntimeError, match="already synchronous"):
        session(world).open()
    assert world.settings.synchronous_mode


def test_startup_failure_restores_partial_settings():
    world = FakeWorld()
    before = vars(world.get_settings())
    world.fail_apply_once = True
    with pytest.raises(RuntimeError, match="apply timeout"):
        session(world).open()
    assert vars(world.get_settings()) == before
    with session(world):
        pass


def test_sensors_disabled_and_unowned_actor_untouched():
    world = FakeWorld()
    with session(world) as owner:
        with pytest.raises(ValueError, match="Sensors"):
            owner.try_spawn(SimpleNamespace(id="sensor.camera.rgb"), object())
    assert world.actor.is_alive
    assert world.destroyed == 0


def test_external_frame_jump_brakes():
    world = FakeWorld()
    with session(world) as owner:
        spawn(owner)
        owner.tick()
        world.frame += 1
        with pytest.raises(RuntimeError, match="Nonconsecutive"):
            owner.tick()
        assert world.controls[-1].brake == 1


def test_wrong_thread_cannot_tick():
    import threading

    world = FakeWorld()
    errors = []
    with session(world) as owner:

        def attempt():
            try:
                owner.tick()
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=attempt)
        thread.start()
        thread.join()
    assert errors and "opening thread" in errors[0]
    assert world.frame == 10


def test_destroy_failure_still_restores_and_releases_owner():
    world = FakeWorld()
    with pytest.raises(RuntimeError, match="Cleanup incomplete"):
        with session(world) as owner:
            spawn(owner)
            world.actor.destroy = lambda: False
    assert not world.settings.synchronous_mode
    with session(world):
        pass


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixed_delta_seconds": 0},
        {"fixed_delta_seconds": float("nan")},
        {"max_substep_delta_time": 0.02},
        {"max_substeps": True},
        {"max_substeps": 0},
        {"max_substeps": 11},
        {"fixed_delta_seconds": 0.1, "max_substeps": 5},
        {"tick_timeout_seconds": -1},
    ],
)
def test_invalid_physics_timing_rejected(kwargs):
    with pytest.raises(ValueError):
        WorldTiming(**kwargs)


class MapClient:
    """Fake client whose load_world times out while the server keeps loading."""

    def __init__(self, current, load_times_out=False, ready_after_polls=2):
        self.current = current
        self.load_times_out = load_times_out
        self.ready_after_polls = ready_after_polls
        self.loading = None
        self.loads = []
        self.timeouts = []

    def set_timeout(self, seconds):
        self.timeouts.append(seconds)

    def _world(self, name):
        return SimpleNamespace(get_map=lambda: SimpleNamespace(name=f"Carla/Maps/{name}"))

    def get_world(self):
        if self.loading is not None:
            self.ready_after_polls -= 1
            if self.ready_after_polls <= 0:
                self.current, self.loading = self.loading, None
        return self._world(self.current)

    def load_world(self, name):
        self.loads.append(name)
        if self.load_times_out:
            self.loading = name
            raise RuntimeError("time-out of 60000ms while waiting for the simulator")
        self.current = name
        return self._world(name)


def test_ensure_map_skips_loading_when_the_map_is_already_there():
    from carla_adapter.world import ensure_map

    client = MapClient("Town03")
    world = ensure_map(client, "Town03", log=lambda text: None)
    assert world.get_map().name.endswith("Town03") and client.loads == []


def test_ensure_map_waits_for_a_load_that_outlives_the_client_timeout():
    from carla_adapter.world import ensure_map

    client = MapClient("Town10HD_Opt", load_times_out=True)
    clock = iter(range(1000))
    world = ensure_map(client, "Town03", log=lambda text: None, sleep=lambda s: None, clock=lambda: next(clock))
    assert world.get_map().name.endswith("Town03")
    assert client.loads == ["Town03"] and client.timeouts[0] == 300.0 and client.timeouts[-1] == 60.0


def test_ensure_map_gives_up_after_its_own_timeout():
    from carla_adapter.world import ensure_map

    client = MapClient("Town10HD_Opt", load_times_out=True, ready_after_polls=10**6)
    clock = iter(range(0, 10**6, 50))
    with pytest.raises(RuntimeError, match="did not load within 300 s"):
        ensure_map(client, "Town03", log=lambda text: None, sleep=lambda s: None, clock=lambda: next(clock))


def test_close_skips_actors_the_server_already_removed():
    world = FakeWorld()
    gone = SimpleNamespace(id=99, type_id="walker.pedestrian.0001", is_alive=True)
    gone.destroy = lambda: (_ for _ in ()).throw(RuntimeError("unable to destroy actor: not found"))
    world.get_actors = lambda: [world.actor]
    with session(world) as owner:
        spawn(owner)
        owner._actors.append(gone)
    assert owner.vanished_actor_ids == [99]
    assert world.destroyed == 1
