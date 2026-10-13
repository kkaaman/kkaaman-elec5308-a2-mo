#!/usr/bin/env python3
"""CARLA machine smoke test for PROC-02 (and early evidence for BL-B-01).

What it does:

1. Records machine details: OS, CPU, RAM, GPU/VRAM/driver, Python, and the CARLA
   client and server versions.
2. Loads Town10HD (skips the reload if it is already the current map).
3. Switches to synchronous mode with a fixed step, spawns one car with co-located
   RGB, depth, semantic and instance-segmentation cameras, and ticks N frames,
   checking that every camera delivers the exact frame that was ticked.
4. Saves one image per camera, destroys everything it spawned, and checks nothing
   was left behind. Repeats spawn/cleanup ``--cycles`` times, then restores the
   world settings it found and checks they were restored.
5. Writes ``runs/smoke/<host>-<time>/smoke.json`` and prints a row for
   ``docs/machines.md``.

Start the CARLA server first, close any other script that ticks the world, then::

    python scripts/carla_smoke_test.py --member Mohammad

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import queue
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import carla
except ImportError:
    sys.exit(
        "The 'carla' Python package is not installed in this Python environment.\n"
        "Install the client that matches your server version, e.g.: pip install carla==0.9.16"
    )

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

SENSOR_BLUEPRINTS = {
    "rgb": "sensor.camera.rgb",
    "depth": "sensor.camera.depth",
    "semantic": "sensor.camera.semantic_segmentation",
    "instance": "sensor.camera.instance_segmentation",
}
REQUIRED_SENSORS = ("rgb", "depth")
SUBSTEP_S = 0.01


# --------------------------------------------------------------------------- machine info


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatusEx()
            status.dwLength = ctypes.sizeof(MemoryStatusEx)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024**3, 1)
        except Exception:  # noqa: BLE001
            return None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024**2, 1)
    except OSError:
        pass
    return None


def _gpus() -> list[dict] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            name, mem_mib, driver = parts
            vram = round(float(mem_mib) / 1024, 1) if mem_mib.replace(".", "").isdigit() else None
            gpus.append({"name": name, "vram_gb": vram, "driver": driver})
    return gpus


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def machine_info() -> dict:
    return {
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "ram_gb": _ram_gb(),
        "gpus": _gpus(),
        "numpy": np.__version__,
        "carla_package": _package_version("carla"),
    }


# --------------------------------------------------------------------------- helpers


def _xyz(version: str) -> str:
    match = re.search(r"\d+\.\d+\.\d+", version or "")
    return match.group(0) if match else (version or "")


def _short_map(name: str) -> str:
    return name.replace("\\", "/").split("/")[-1]


def _map_matches(name: str, wanted: str) -> bool:
    short = _short_map(name)
    return short in (wanted, f"{wanted}_Opt")


def _settings_snapshot(settings) -> dict:
    return {
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "substepping": bool(settings.substepping),
        "max_substep_delta_time": settings.max_substep_delta_time,
        "max_substeps": settings.max_substeps,
        "no_rendering_mode": bool(settings.no_rendering_mode),
    }


def _close(a, b, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def _percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100 * len(ordered)) - 1))
    return ordered[index]


def _bgra(image) -> np.ndarray:
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)


def depth_metres(image) -> np.ndarray:
    """Decode a CARLA depth image (BGRA bytes) to metres, avoiding uint8 overflow."""
    bgra = _bgra(image).astype(np.float64)
    b, g, r = bgra[..., 0], bgra[..., 1], bgra[..., 2]
    return 1000.0 * (r + 256.0 * g + 65536.0 * b) / 16777215.0


def wait_for_frame(sensor_queue: queue.Queue, frame: int, timeout_s: float):
    """Return (data, status, stale_count) for the measurement matching ``frame``."""
    deadline = time.perf_counter() + timeout_s
    stale = 0
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None, "timeout", stale
        try:
            data = sensor_queue.get(timeout=remaining)
        except queue.Empty:
            return None, "timeout", stale
        if data.frame == frame:
            return data, "ok", stale
        if data.frame < frame:
            stale += 1
            continue
        return data, "ahead", stale


def _choose_vehicle(blueprints, wanted: str):
    matches = blueprints.filter(wanted)
    if matches:
        return matches[0], None
    for bp in blueprints.filter("vehicle.*"):
        if (
            bp.has_attribute("number_of_wheels")
            and bp.get_attribute("number_of_wheels").as_int() == 4
        ):
            return bp, f"{wanted} not found; used {bp.id}"
    raise RuntimeError("no four-wheel vehicle blueprint available")


# --------------------------------------------------------------------------- one cycle


def run_cycle(world, args, cycle: int, out_dir: Path, save_images: bool) -> dict:
    """Spawn car + cameras, tick, check frames, clean up. Always destroys what it spawned."""
    result: dict = {"cycle": cycle, "warnings": []}
    blueprints = world.get_blueprint_library()
    spawned = []
    sensors: dict = {}
    try:
        vehicle_bp, note = _choose_vehicle(blueprints, args.vehicle)
        if note:
            result["warnings"].append(note)
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "hero")
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("map has no spawn points")
        start = args.spawn_index % len(spawn_points)
        vehicle = None
        used_spawn = None
        for spawn_point in spawn_points[start:] + spawn_points[:start]:
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                used_spawn = spawn_point
                break
        if vehicle is None:
            raise RuntimeError("could not spawn the vehicle at any spawn point")
        spawned.append(vehicle)
        result["vehicle"] = vehicle.type_id

        camera_tf = carla.Transform(carla.Location(x=1.5, z=1.7))
        for kind, bp_id in SENSOR_BLUEPRINTS.items():
            found = blueprints.filter(bp_id)
            if not found:
                if kind in REQUIRED_SENSORS:
                    raise RuntimeError(f"required sensor blueprint {bp_id} missing")
                result["warnings"].append(f"{bp_id} not available in this CARLA version")
                continue
            bp = found[0]
            bp.set_attribute("image_size_x", str(args.width))
            bp.set_attribute("image_size_y", str(args.height))
            bp.set_attribute("fov", str(args.fov))
            if bp.has_attribute("sensor_tick"):
                bp.set_attribute("sensor_tick", "0.0")
            actor = world.spawn_actor(bp, camera_tf, attach_to=vehicle)
            spawned.append(actor)
            sensor_queue: queue.Queue = queue.Queue()
            actor.listen(sensor_queue.put)
            sensors[kind] = (actor, sensor_queue)

        if cycle == 1:  # put the editor view behind the car so you can watch it
            fwd = used_spawn.get_forward_vector()
            loc = used_spawn.location
            world.get_spectator().set_transform(
                carla.Transform(
                    carla.Location(x=loc.x - 8.0 * fwd.x, y=loc.y - 8.0 * fwd.y, z=loc.z + 4.0),
                    carla.Rotation(pitch=-15.0, yaw=used_spawn.rotation.yaw),
                )
            )

        def zero_counts():
            return {k: {"ok": 0, "timeout": 0, "ahead": 0, "stale_dropped": 0} for k in sensors}

        counts, warmup_counts = zero_counts(), zero_counts()
        frame_ms: list[float] = []
        last_ok: dict = {}
        for tick in range(args.ticks):
            t0 = time.perf_counter()
            frame = world.tick(args.tick_timeout)
            tick_data = {}
            bucket = counts if tick >= args.warmup else warmup_counts
            for kind, (_, sensor_queue) in sensors.items():
                data, status, stale = wait_for_frame(sensor_queue, frame, args.sensor_timeout)
                bucket[kind][status] += 1
                bucket[kind]["stale_dropped"] += stale
                if status == "ok":
                    tick_data[kind] = data
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if tick >= args.warmup:
                frame_ms.append(elapsed_ms)
            if len(tick_data) == len(sensors):
                last_ok = tick_data

        result["sensor_counts"] = counts
        result["warmup_sensor_counts"] = warmup_counts
        result["frames_ok"] = all(
            c["ok"] > 0 and c["timeout"] == 0 and c["ahead"] == 0 for c in counts.values()
        )
        if frame_ms:
            mean_ms = statistics.fmean(frame_ms)
            result["frame_ms"] = {
                "mean": round(mean_ms, 1),
                "p50": round(_percentile(frame_ms, 50), 1),
                "p95": round(_percentile(frame_ms, 95), 1),
                "max": round(max(frame_ms), 1),
                "n": len(frame_ms),
            }
            result["real_time_factor"] = round(args.dt / (mean_ms / 1000.0), 2)

        if "depth" in last_ok:
            depth = depth_metres(last_ok["depth"])
            h, w = depth.shape
            centre = depth[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
            result["depth_m"] = {
                "min": round(float(depth.min()), 2),
                "centre_median": round(float(np.median(centre)), 2),
                "max": round(float(depth.max()), 2),
            }
            result["depth_ok"] = bool(
                np.isfinite(depth).all() and 0.0 < float(np.median(centre)) <= 1000.0
            )
        else:
            result["depth_ok"] = False
        if "semantic" in last_ok:
            tags = np.unique(_bgra(last_ok["semantic"])[..., 2])
            result["semantic_tags_seen"] = [int(t) for t in tags]
        if "instance" in last_ok:
            bgra = _bgra(last_ok["instance"])
            ids = np.unique(bgra[..., 1].astype(np.int32) * 256 + bgra[..., 0])
            result["instance_ids_seen"] = int(len(ids))

        if save_images and last_ok:
            converters = {
                "rgb": carla.ColorConverter.Raw,
                "depth": carla.ColorConverter.LogarithmicDepth,
                "semantic": carla.ColorConverter.CityScapesPalette,
                "instance": carla.ColorConverter.Raw,
            }
            for kind, data in last_ok.items():
                data.save_to_disk(str(out_dir / f"{kind}.png"), converters[kind])
            result["images_saved"] = sorted(last_ok)
    finally:
        ids = [a.id for a in spawned]
        for actor, _ in sensors.values():
            try:
                actor.stop()
            except Exception:  # noqa: BLE001
                pass
        destroy_failed = []
        for actor in reversed(spawned):
            try:
                if not actor.destroy():
                    destroy_failed.append(actor.id)
            except Exception as exc:  # noqa: BLE001
                destroy_failed.append(actor.id)
                result["warnings"].append(f"destroy {actor.id} failed: {exc}")
        result["destroy_failed_ids"] = destroy_failed
        try:
            world.tick(args.tick_timeout)
            # get_actors(ids) can answer from the client's actor cache, which keeps
            # destroyed actors; the full list is built from the current world snapshot.
            alive = {a.id for a in world.get_actors()}
            leftovers = [i for i in ids if i in alive]
        except Exception as exc:  # noqa: BLE001
            leftovers = ids
            result["warnings"].append(f"could not confirm cleanup: {exc}")
        result["spawned_actor_ids"] = ids
        result["leftover_actor_ids"] = leftovers
        result["cleanup_ok"] = not leftovers
    return result


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CARLA machine smoke test (PROC-02)")
    parser.add_argument("--member", default="", help="your name, for the machines table")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=20.0, help="client RPC timeout (s)")
    parser.add_argument("--map", default="Town10HD")
    parser.add_argument("--vehicle", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.10, help="fixed simulation step (s)")
    parser.add_argument("--ticks", type=int, default=100, help="ticks per cycle")
    parser.add_argument("--warmup", type=int, default=10, help="ticks excluded from timing")
    parser.add_argument("--cycles", type=int, default=2, help="spawn/cleanup repetitions")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--tick-timeout", type=float, default=10.0)
    parser.add_argument("--sensor-timeout", type=float, default=2.0)
    parser.add_argument("--out", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--no-images", action="store_true")
    args = parser.parse_args(argv)

    if args.dt > SUBSTEP_S * 16:
        parser.error(f"--dt must be <= {SUBSTEP_S * 16:.2f} s for CARLA physics substepping")

    started = datetime.now(timezone.utc)
    report: dict = {
        "task": "PROC-02 smoke test",
        "member": args.member,
        "started_utc": started.isoformat(timespec="seconds"),
        "machine": machine_info(),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "checks": {},
        "warnings": [],
    }
    host = report["machine"]["host"]
    out_dir = args.out / f"{host}-{started.strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checks = report["checks"]

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = None
    original = None
    try:
        report["carla_client"] = client.get_client_version()
        report["carla_server"] = client.get_server_version()
        checks["versions_match"] = _xyz(report["carla_client"]) == _xyz(report["carla_server"])
        print(f"Connected: client {report['carla_client']}, server {report['carla_server']}")

        world = client.get_world()
        current = world.get_map().name
        if _map_matches(current, args.map):
            report["map_loaded_by_script"] = False
        else:
            available = [_short_map(m) for m in client.get_available_maps()]
            target = next((m for m in (args.map, f"{args.map}_Opt") if m in available), None)
            if target is None:
                raise RuntimeError(
                    f"{args.map} not available. Maps: {', '.join(sorted(available))}"
                )
            print(f"Loading {target} (can take a minute)...")
            client.set_timeout(max(args.timeout, 180.0))
            t0 = time.perf_counter()
            world = client.load_world(target)
            client.set_timeout(args.timeout)
            report["map_load_s"] = round(time.perf_counter() - t0, 1)
            report["map_loaded_by_script"] = True
        report["map"] = _short_map(world.get_map().name)
        checks["map_ok"] = _map_matches(report["map"], args.map)

        original = world.get_settings()
        report["original_settings"] = _settings_snapshot(original)
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.dt
        settings.substepping = True
        settings.max_substep_delta_time = SUBSTEP_S
        settings.max_substeps = max(1, min(16, math.ceil(round(args.dt / SUBSTEP_S, 6))))
        world.apply_settings(settings)
        applied = _settings_snapshot(world.get_settings())
        report["applied_settings"] = applied
        checks["sync_applied"] = applied["synchronous_mode"] and _close(
            applied["fixed_delta_seconds"], args.dt
        )
        world.tick(args.tick_timeout)

        cycles = []
        for cycle in range(1, args.cycles + 1):
            print(f"Cycle {cycle}/{args.cycles}: spawning car and cameras, {args.ticks} ticks...")
            cycles.append(run_cycle(world, args, cycle, out_dir, not args.no_images and cycle == 1))
        report["cycles"] = cycles
        checks["frames_ok"] = all(c.get("frames_ok", False) for c in cycles)
        checks["depth_ok"] = all(c.get("depth_ok", False) for c in cycles)
        checks["cleanup_ok"] = all(c.get("cleanup_ok", False) for c in cycles)
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if world is not None and original is not None:
            try:
                world.apply_settings(original)
                restored = _settings_snapshot(world.get_settings())
                report["restored_settings"] = restored
                want = report["original_settings"]
                checks["settings_restored"] = restored["synchronous_mode"] == want[
                    "synchronous_mode"
                ] and _close(restored["fixed_delta_seconds"], want["fixed_delta_seconds"])
            except Exception as exc:  # noqa: BLE001
                checks["settings_restored"] = False
                report["warnings"].append(f"could not restore world settings: {exc}")

    expected = (
        "versions_match", "map_ok", "sync_applied", "frames_ok",
        "depth_ok", "cleanup_ok", "settings_restored",
    )  # fmt: skip
    for name in expected:
        checks.setdefault(name, False)
    report["passed"] = "error" not in report and all(checks.values())
    report["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (out_dir / "smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # ---- console summary
    print()
    for name in expected:
        print(f"  [{'PASS' if checks[name] else 'FAIL'}] {name}")
    if "error" in report:
        print(f"  error: {report['error']}")
    for cycle in report.get("cycles", []):
        timing = cycle.get("frame_ms", {})
        print(
            f"  cycle {cycle['cycle']}: frame p95 {timing.get('p95', '?')} ms, "
            f"real-time factor {cycle.get('real_time_factor', '?')}, "
            f"depth centre median {cycle.get('depth_m', {}).get('centre_median', '?')} m"
        )
        for kind, c in cycle.get("sensor_counts", {}).items():
            if c["timeout"] or c["ahead"]:
                print(
                    f"    {kind} camera: {c['timeout']} timeouts, {c['ahead']} out-of-order "
                    f"frames out of {c['ok'] + c['timeout'] + c['ahead']} measured ticks"
                )
        if cycle.get("leftover_actor_ids"):
            print(f"    actors left behind after cleanup: {cycle['leftover_actor_ids']}")
        for warning in cycle.get("warnings", []):
            print(f"  warning: {warning}")
    for warning in report["warnings"]:
        print(f"  warning: {warning}")
    print(f"\nReport: {out_dir / 'smoke.json'}")

    m = report["machine"]
    gpu = (
        "; ".join(
            f"{g['name']} ({g['vram_gb']} GB, driver {g['driver']})" for g in (m["gpus"] or [])
        )
        or "not detected"
    )
    first = (report.get("cycles") or [{}])[0]
    row = (
        f"| {args.member or '?'} | {m['host']} | {m['os']} | {m['cpu']} | {m['ram_gb']} GB | {gpu} "
        f"| {m['python']} | {report.get('carla_client', '?')} / {report.get('carla_server', '?')} "
        f"| {report.get('map', '?')} | {first.get('frame_ms', {}).get('p95', '?')} ms @ dt {args.dt} "
        f"| {first.get('real_time_factor', '?')} | {'PASS' if report['passed'] else 'FAIL'} "
        f"| {started.date().isoformat()} |"
    )
    print("\nRow for docs/machines.md:")
    print(
        "| Member | Host | OS | CPU | RAM | GPU | Python | CARLA client / server | Map "
        "| Frame p95 | Real-time factor | Smoke test | Date |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    print(row)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
