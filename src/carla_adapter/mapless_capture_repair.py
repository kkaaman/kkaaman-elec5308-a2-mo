"""Repair schema-1 mapless capture runs: place truth boxes on their bounding-box centres.

Schema 1 (``group7-mapless-capture-1``) wrote each actor's ``x_m, y_m`` from the actor
transform and ignored ``bounding_box.location`` x/y, so most CARLA vehicles' boxes sit
0.1-0.5 m off their true centre (review of 94b5caa). The extent and yaw were right.

The box offset depends only on the blueprint, so this repair needs, per ``type_id``, the
box location in the actor frame. It gets that either from a running CARLA server, by
spawning each type once (``--host``; any map, the ego's own map is not needed), or from
an offsets file written by an earlier run (``--offsets``). With ``--export-offsets`` the
measured offsets are saved for other machines.

The run keeps its data: ``samples.jsonl`` is rewritten with corrected centres and the
plan-view corners (as schema 2 writes them); the original is kept as
``samples.schema1.jsonl``; the manifest records the repair and moves to schema 2.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

SCHEMA_1 = "group7-mapless-capture-1"
SCHEMA_2 = "group7-mapless-capture-2"


def box_offsets_from_carla(type_ids, host, port):
    """{type_id: [dx, dy, dz]} of the bounding-box centre in the actor frame (core axes)."""
    import carla

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    offsets = {}
    for i, type_id in enumerate(sorted(type_ids)):
        bp = library.find(type_id)
        actor = None
        for spawn in spawn_points[i % len(spawn_points) :] + spawn_points[: i % len(spawn_points)]:
            spawn.location.z += 0.5
            actor = world.try_spawn_actor(bp, spawn)
            if actor is not None:
                break
        if actor is None:
            raise RuntimeError(f"Could not spawn {type_id} to read its bounding box")
        try:
            loc = actor.bounding_box.location
            offsets[type_id] = [float(loc.x), -float(loc.y), float(loc.z)]
        finally:
            actor.destroy()
    return offsets


def repair_actor(actor, offset):
    dx, dy, dz = offset
    c, s = math.cos(actor["yaw_rad"]), math.sin(actor["yaw_rad"])
    x = actor["x_m"] + c * dx - s * dy
    y = actor["y_m"] + s * dx + c * dy
    ex, ey = actor["extent_m"][0], actor["extent_m"][1]
    corners = [[x + c * u * ex - s * v * ey, y + s * u * ex + c * v * ey] for u, v in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
    # Schema 1 already added box.location.z to z_m.
    return dict(actor, x_m=x, y_m=y, corners_m=corners, schema1_x_m=actor["x_m"], schema1_y_m=actor["y_m"])


def repair_run(run, offsets):
    run = Path(run)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_1:
        raise ValueError(f"{run} is {manifest.get('schema')}, not {SCHEMA_1}: nothing to repair")
    samples_path = run / "samples.jsonl"
    backup = run / "samples.schema1.jsonl"
    if not backup.exists():
        shutil.copy2(samples_path, backup)
    lines = backup.read_text(encoding="utf-8").splitlines()
    missing = set()
    out = []
    for line in lines:
        if not line:
            continue
        sample = json.loads(line)
        actors = []
        for actor in sample["truth"].get("actors", []):
            offset = offsets.get(actor["type_id"])
            if offset is None:
                missing.add(actor["type_id"])
                actors.append(actor)
            else:
                actors.append(repair_actor(actor, offset))
        sample["truth"]["actors"] = actors
        out.append(json.dumps(sample))
    if missing:
        raise RuntimeError(f"No box offset for {sorted(missing)}; nothing written")
    samples_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    manifest.update(
        schema=SCHEMA_2,
        box_geometry="centre and corners repaired from per-blueprint bounding_box.location (schema 1 capture)",
        repaired_from=SCHEMA_1,
        box_offsets=offsets,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return len(out)


def type_ids_in(run):
    ids = set()
    for line in (Path(run) / "samples.jsonl").read_text(encoding="utf-8").splitlines():
        if line:
            for actor in json.loads(line)["truth"].get("actors", []):
                ids.add(actor["type_id"])
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, action="append", help="A schema-1 run folder (repeatable)")
    parser.add_argument("--offsets", type=Path, help="JSON {type_id: [dx, dy, dz]} from an earlier --export-offsets")
    parser.add_argument("--export-offsets", type=Path, help="Write the offsets used (merged) to this JSON file")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--no-carla", action="store_true", help="Fail instead of asking CARLA for missing offsets")
    args = parser.parse_args(argv)
    offsets = json.loads(args.offsets.read_text(encoding="utf-8")) if args.offsets else {}
    needed = set()
    for run in args.run:
        needed |= type_ids_in(run)
    missing = needed - set(offsets)
    if missing:
        if args.no_carla:
            parser.error(f"Offsets file lacks {sorted(missing)}")
        offsets.update(box_offsets_from_carla(missing, args.host, args.port))
    if args.export_offsets:
        args.export_offsets.write_text(json.dumps(offsets, indent=2, sort_keys=True), encoding="utf-8")
    for run in args.run:
        n = repair_run(run, {k: offsets[k] for k in type_ids_in(run)})
        print(json.dumps(dict(run=str(run), samples=n, status="repaired")), file=sys.stderr)


if __name__ == "__main__":
    main()
