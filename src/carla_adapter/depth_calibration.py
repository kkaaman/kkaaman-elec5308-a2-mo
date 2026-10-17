"""Measure raw depth convention against off-axis geometry ray intersections."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from carla_adapter.conversions import decode_depth
from carla_adapter.dataset_capture import FrameBuffer
from carla_adapter.world import SynchronousWorldSession, _settings_record


def ray_reference(world, camera_transform, u, v, width, height, fov, limit=100.0):
    import carla

    focal = width / (2 * math.tan(math.radians(fov) / 2))
    local = np.array([1.0, (u - width / 2) / focal, -(v - height / 2) / focal])
    matrix = np.asarray(camera_transform.get_matrix())
    origin = matrix[:3, 3]
    direction = matrix[:3, :3] @ local
    end = origin + limit * direction / np.linalg.norm(direction)
    hits = world.cast_ray(carla.Location(*map(float, origin)), carla.Location(*map(float, end)))
    if not hits:
        return None
    hit = hits[0].location
    point = np.array([hit.x, hit.y, hit.z])
    relative = point - origin
    axial = float((matrix[:3, :3].T @ relative)[0])
    radial = float(np.linalg.norm(relative))
    return dict(
        axial_m=axial, radial_m=radial, semantic=str(hits[0].label), world_xyz_m=point.tolist()
    )


def choose_convention(rows):
    valid = [
        r
        for r in rows
        if r.get("reference")
        and r.get("raw_depth_m") is not None
        and 5 <= r["reference"]["radial_m"] <= 80
        and r["reference"]["radial_m"] > r["reference"]["axial_m"] * 1.1
    ]
    scores = {
        name: float(np.median([abs(r["raw_depth_m"] - r["reference"][name + "_m"]) for r in valid]))
        if valid
        else None
        for name in ("axial", "radial")
    }
    selected = None
    if len(valid) >= 20:
        best = min(scores, key=scores.get)
        other = "radial" if best == "axial" else "axial"
        if scores[best] <= 0.3 and scores[other] >= max(0.6, 3 * scores[best]):
            selected = best
    return dict(
        status="MEASURED_PASS" if selected else "INCONCLUSIVE",
        convention=selected,
        off_axis_samples=len(valid),
        median_absolute_error_m=scores,
        comparison="raw sensor depth versus same-ray geometry hit; render/collision meshes can differ",
    )


def collect(client, output):
    import carla

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    world = client.get_world()
    before = _settings_record(world.get_settings())
    ids = {a.id for a in world.get_actors()}
    if any(a.type_id.startswith(("vehicle.", "walker.", "sensor.")) for a in world.get_actors()):
        raise RuntimeError("Use empty world")
    if client.get_server_version() != "0.9.16" or client.get_client_version() != "0.9.16":
        raise RuntimeError("Matching CARLA 0.9.16 required")
    rows = []
    report = {}
    error = None
    try:
        with SynchronousWorldSession(client) as owner:
            settings = world.get_settings()
            settings.no_rendering_mode = False
            world.apply_settings(settings)
            points = world.get_map().get_spawn_points()
            for site in (0, min(70, len(points) - 1), min(152, len(points) - 1)):
                vehicle = owner.try_spawn(
                    world.get_blueprint_library().find("vehicle.tesla.model3"), points[site]
                )
                if vehicle is None:
                    raise RuntimeError("Calibration spawn occupied")
                vehicle.apply_control(carla.VehicleControl(brake=1))
                for _ in range(20):
                    owner.tick()
                bp = world.get_blueprint_library().find("sensor.camera.depth")
                for key, value in {
                    "image_size_x": 640,
                    "image_size_y": 360,
                    "fov": 90,
                    "sensor_tick": 0,
                    "lens_k": 0,
                    "lens_kcube": 0,
                }.items():
                    bp.set_attribute(key, str(value))
                sensor = world.spawn_actor(
                    bp,
                    carla.Transform(carla.Location(x=1.5, z=2)),
                    attach_to=vehicle,
                    attachment_type=carla.AttachmentType.Rigid,
                )
                owner._actors.append(sensor)
                frames = FrameBuffer(("depth",))
                sensor.listen(lambda image: frames.put("depth", image))
                for _ in range(18):
                    last = owner.tick()
                    time.sleep(0.005)
                image = frames.take(last - 4)["depth"]
                depth = decode_depth(image.raw_data, 640, 360)
                for u in (35, 75, 115, 155, 195, 445, 485, 525, 565, 605):
                    for v in (170, 200, 230, 260, 290, 320):
                        reference = ray_reference(world, image.transform, u, v, 640, 360, 90)
                        value = float(depth[v, u])
                        value = value if math.isfinite(value) else None
                        rows.append(
                            dict(
                                site=site,
                                frame_id=image.frame,
                                u=u,
                                v=v,
                                raw_depth_m=value,
                                reference=reference,
                            )
                        )
                sensor.stop()
                sensor.destroy()
                vehicle.destroy()
            report = choose_convention(rows)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    restored = _settings_record(world.get_settings()) == before
    cleaned = {a.id for a in world.get_actors()} == ids
    report.update(
        server=client.get_server_version(),
        map=world.get_map().name.rsplit("/", 1)[-1],
        opendrive_sha256=hashlib.sha256(
            world.get_map().to_opendrive().replace("\r\n", "\n").encode()
        ).hexdigest(),
        rig={"width": 640, "height": 360, "fov_deg": 90, "x_actor_m": 1.5, "z_actor_m": 2.0},
        rows=rows,
        error=error,
        settings_restored=restored,
        no_actor_leaks=cleaned,
    )
    if error or not restored or not cleaned:
        report["status"] = "INCOMPLETE"
    (output / "depth_calibration.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if report.get("status") != "MEASURED_PASS":
        raise RuntimeError("Depth convention unresolved; inspect saved raw measurements")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    args = parser.parse_args()
    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(60)
    collect(client, args.output)
