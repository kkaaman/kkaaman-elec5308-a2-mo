"""Prepare or execute the paired 27-cell stationary dataset with the existing collector.

This is sensor/detector dataset evidence, never 27 driving acceptance runs.
Default is PLAN ONLY and never imports CARLA or opens a connection.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from scripts.prepare_weather_matrix import prepare


def combine_captures(output, cells, specification_sha256):
    """One shared index references original images; no binary copies."""
    import hashlib

    from evaluation.dataset import read_dataset

    rows, manifests = [], []
    for cell in cells:
        identity = cell["cell_id"]
        manifest, samples = read_dataset(output / identity)
        if manifest["weather_experiment"]["cell_id"] != identity:
            raise ValueError("Capture cell identity mismatch")
        if manifest["weather_experiment"]["specification_sha256"] != specification_sha256:
            raise ValueError("Capture specification mismatch")
        if not manifest.get("settings_restored") or not manifest.get("no_actor_leaks"):
            raise ValueError("Unclean capture cannot enter combined evidence")
        manifests.append(manifest)
        for source in samples:
            row = dict(source)
            for key in ("sample_id", "run_id", "sequence_id", "rgb", "lane", "instance", "depth"):
                row[key] = identity + "/" + row[key]
            rows.append(row)
    if not rows:
        raise ValueError("No captures")
    for manifest in manifests[1:]:
        for key in ("camera", "image_size_wh", "opendrive_sha256", "label_tag"):
            if manifest[key] != manifests[0][key]:
                raise ValueError("Incompatible capture geometry/map/labels")
    index = output / "samples.jsonl"
    with index.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    result = dict(
        manifests[0],
        sample_count=len(rows),
        requested_samples=len(rows),
        split_counts=dict(Counter(row["split"] for row in rows)),
        seed=None,
        paired_seeds=sorted({m["seed"] for m in manifests if m.get("seed") is not None}),
        samples_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
        combined_cells=[c["cell_id"] for c in cells],
        specification_sha256=specification_sha256,
        weather_readbacks={
            c["weather_experiment"]["cell_id"]: c["weather_readback"] for c in manifests
        },
    )
    result["weather_experiment"] = None
    result.pop("weather_readback", None)
    with (output / "manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    read_dataset(output)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/weather/robustness_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--samples-per-cell", type=int, default=600)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate/reuse completed cells and archive failed attempts",
    )
    args = parser.parse_args()
    if args.samples_per_cell < 200:
        parser.error("At least 200 samples/cell required")
    matrix = prepare(args.config)
    cells = matrix["cells"]
    if args.cells:
        unknown = set(args.cells) - {c["cell_id"] for c in cells}
        if unknown or len(set(args.cells)) != len(args.cells):
            parser.error("Unknown or duplicate cells")
        cells = [c for c in cells if c["cell_id"] in args.cells]
    previous = None
    if args.resume:
        if not args.execute:
            parser.error("--resume requires --execute")
        previous = json.loads((args.output / "queue.json").read_text())
        if previous["specification_sha256"] != matrix["specification_sha256"]:
            parser.error("Cannot resume a different weather specification")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    commands = []
    for cell in cells:
        command = [
            sys.executable,
            "-m",
            "carla_adapter.dataset_capture",
            "--weather-config",
            str(args.config.resolve()),
            "--weather-cell",
            cell["cell_id"],
            "--host",
            args.host,
            "--output",
            str((args.output / cell["cell_id"]).resolve()),
            "--samples",
            str(args.samples_per_cell),
            "--rounds",
            "1",
            "--sequence-frames",
            "20",
        ]
        commands.append(dict(cell_id=cell["cell_id"], command=command, status="NOT_RUN"))
    if previous is not None:
        from evaluation.dataset import read_dataset

        old_cells = {c["cell_id"]: c for c in previous["cells"]}
        if set(old_cells) != {c["cell_id"] for c in commands}:
            parser.error("Resume must retain the original cell set")
        history = args.output / "attempt-history"
        history.mkdir(exist_ok=True)
        attempt = 1
        while (history / f"queue-{attempt:03d}.json").exists():
            attempt += 1
        (history / f"queue-{attempt:03d}.json").write_text(json.dumps(previous, indent=2) + "\\n")
        for item in commands:
            old = old_cells[item["cell_id"]]
            if old["command"] != item["command"]:
                parser.error("Resume command/configuration differs")
            directory = args.output / item["cell_id"]
            if old["status"] == "COLLECTED":
                manifest, _ = read_dataset(directory)
                if (
                    manifest["sample_count"] != args.samples_per_cell
                    or not manifest["settings_restored"]
                    or not manifest["no_actor_leaks"]
                ):
                    raise ValueError("Completed cell fails resume validation")
                item.update(old, reused=True)
                continue
            if directory.exists():
                archive = history / f"{item['cell_id']}-attempt-{attempt:03d}"
                archive.mkdir()
                source = directory.resolve()
                if not source.is_relative_to(args.output.resolve()):
                    raise ValueError("Capture path outside evidence root")
                directory.rename(archive / "capture")
                log = args.output / (item["cell_id"] + ".log")
                if log.exists():
                    log.rename(archive / "capture.log")
                (archive / "queue-entry.json").write_text(json.dumps(old, indent=2) + "\\n")

    report = dict(
        status="PLAN_ONLY",
        specification_sha256=matrix["specification_sha256"],
        scope="Stationary paired sensor data, not baseline driving acceptance",
        cells=commands,
    )

    def save():
        (args.output / "queue.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    if args.execute:
        report["status"] = "CAPTURING"
        for item in commands:
            if item["status"] == "COLLECTED":
                continue
            item["source_revision"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip()
            with (args.output / (item["cell_id"] + ".log")).open("x") as stream:
                result = subprocess.run(
                    item["command"], cwd=root, stdout=stream, stderr=subprocess.STDOUT
                )
            item.update(
                returncode=result.returncode,
                status="COLLECTED" if result.returncode == 0 else "FAILED",
            )
            save()
            if result.returncode:
                report["status"] = "INCOMPLETE"
                save()
                return 1
        if (args.output / "samples.jsonl").exists():
            for name in ("samples.jsonl", "manifest.json"):
                path = args.output / name
                if path.exists():
                    path.rename(args.output / "attempt-history" / f"{name}.before-recombine")
        combine_captures(args.output, cells, matrix["specification_sha256"])
        report["status"] = "COLLECTED_REQUIRES_REVIEW"
        save()
    print(report["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
