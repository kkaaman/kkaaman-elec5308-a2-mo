"""Parked BL-B-02 alignment/timeout smoke check; no depth-accuracy claim."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from autonomy.interfaces import EgoPose
from carla_adapter.sensors import RGBDCapture
from carla_adapter.vehicle import FixedVehicleAdapter
from carla_adapter.world import SynchronousWorldSession, _settings_record


def prepare_parked_adapter(owner, vehicle, brake_control):
    """Advance parked physics before querying world-space wheel positions."""
    vehicle.apply_control(brake_control)
    for _ in range(20):
        owner.tick()
    return FixedVehicleAdapter(vehicle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    import carla

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(30)
    world = client.get_world()
    before = _settings_record(world.get_settings())
    ids = {
        a.id
        for a in world.get_actors()
        if a.type_id.startswith(("vehicle.", "sensor."))
    }
    if ids:
        raise RuntimeError(
            "Stop other driving clients; existing vehicles/sensors present"
        )
    if not world.get_map().name.endswith("Town10HD_Opt"):
        raise RuntimeError("Expected Town10HD_Opt")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="INCOMPLETE",
        server=client.get_server_version(),
        map=world.get_map().name,
        scope="parked exact-frame alignment and injected sensor timeout; raw depth convention/accuracy unverified",
        frames=[],
    )
    failure = None
    try:
        with SynchronousWorldSession(client) as owner:
            vehicle = owner.try_spawn(
                world.get_blueprint_library().find("vehicle.tesla.model3"),
                world.get_map().get_spawn_points()[152],
            )
            if vehicle is None:
                raise RuntimeError("Spawn 152 occupied")
            adapter = prepare_parked_adapter(
                owner, vehicle, carla.VehicleControl(brake=1.0, throttle=0.0)
            )
            with RGBDCapture(owner, vehicle, "rgbd-smoke") as capture:
                report["rig"] = asdict(capture.rig)
                frames = []
                for index in range(32):
                    frame = owner.tick()
                    frames.append(frame)
                    # In a fixed-step world no further tick can alter this pose.
                    capture.record_snapshot(
                        frame,
                        EgoPose(adapter.state(), vehicle.get_transform().location.z),
                    )
                    if index >= 8:
                        matched = capture.read(frames[index - 4], 5)
                        report["frames"].append(
                            dict(
                                frame=matched.frame_id,
                                time_s=matched.sim_time_s,
                                ego=asdict(matched.ego),
                                rgb_shape=list(matched.rgb.shape),
                                depth_shape=list(matched.decoded_depth_m.shape),
                            )
                        )
                        if index == 8:
                            np.savez_compressed(
                                args.output / "sample.npz",
                                rgb=matched.rgb,
                                decoded_depth_m=matched.decoded_depth_m,
                            )
                try:
                    capture.read(frames[-1] + 100, 0.1)
                except TimeoutError:
                    control = vehicle.get_control()
                    report["timeout_brake_pass"] = (
                        control.brake == 1 and control.throttle == 0
                    )
                else:
                    raise RuntimeError("Missing-frame injection unexpectedly succeeded")
                report["queue_events"] = capture.joiner.stats()
                report["callback_errors"] = capture.callback_errors
                if capture.callback_errors or not report["timeout_brake_pass"]:
                    raise RuntimeError("Callback failure or timeout stop not applied")
        report["status"] = "PASS"
    except BaseException as error:
        failure = error
        report["failure"] = f"{type(error).__name__}: {error}"
    finally:
        report["settings_restored"] = _settings_record(world.get_settings()) == before
        report["no_actor_leaks"] = {
            a.id
            for a in world.get_actors()
            if a.type_id.startswith(("vehicle.", "sensor."))
        } == ids
        if not report["settings_restored"] or not report["no_actor_leaks"]:
            report["status"] = "INCOMPLETE"
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("frames", "rig")},
                indent=2,
            )
        )
        print("Matched frames:", len(report["frames"]))
        print("Saved:", args.output)
    if failure is not None:
        raise failure
    if report["status"] != "PASS":
        raise RuntimeError("Cleanup verification failed")


if __name__ == "__main__":
    main()
