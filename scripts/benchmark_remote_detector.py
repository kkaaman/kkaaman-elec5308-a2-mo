"""E4-07/09 reproducible RGB .npy replay latency sweep (no simulator).

Local factory syntax: package.module:callable returning a contract Detector.
Use the same model and hardware notes for comparable modes. This script never
calls fixture timings learned-model performance and does not infer accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import statistics
import time
from pathlib import Path

import numpy as np

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle
from navigation_control.types import VehicleState
from perception.async_detector import matches, positive
from perception.remote_detector import JpegHttpBackend
from perception.replay_accuracy import match_counts


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered)-1)*fraction
    lo = int(at)
    return ordered[lo] + (ordered[min(lo+1, len(ordered)-1)]-ordered[lo])*(at-lo)


def summarize(rows, deadline_s):
    successful = [r["total_s"] for r in rows if r["status"] == "OK"]
    counts = {key: sum(r.get(key) or 0 for r in rows) for key in ("tp", "fp", "fn")}
    annotated = bool(rows) and all(r.get("tp") is not None for r in rows)
    accuracy = dict(status="MEASURED" if annotated else "NOT_MEASURED", **counts,
                    precision=counts["tp"]/(counts["tp"]+counts["fp"]) if annotated and counts["tp"]+counts["fp"] else None,
                    recall=counts["tp"]/(counts["tp"]+counts["fn"]) if annotated and counts["tp"]+counts["fn"] else None)
    return dict(accuracy=accuracy, frames=len(rows), successful_frames=len(successful),
                failure_count=len(rows)-len(successful),
                p50_s=statistics.median(successful) if successful else None,
                p95_s=percentile(successful, .95), max_s=max(successful, default=None),
                deadline_miss_rate=sum(r["status"] != "OK" or r["total_s"] > deadline_s
                                       for r in rows)/len(rows) if rows else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True, type=Path, help="Directory of >=200 RGB uint8 .npy frames")
    parser.add_argument("--annotations", type=Path, help="JSON mapping every filename to class_name/xyxy truth list")
    parser.add_argument("--output", required=True, type=Path, help="New evidence directory")
    parser.add_argument("--local-factory", required=True, help="module:factory; no arguments")
    parser.add_argument("--lan-endpoint")
    parser.add_argument("--cloud-endpoint")
    parser.add_argument("--model-version", required=True, help="Identical model/weights identity across modes")
    parser.add_argument("--hardware-notes", required=True, help="Client/server CPU/GPU, runtime and network notes")
    parser.add_argument("--origin", required=True, choices=["recorded-model-replay", "synthetic-smoke"])
    parser.add_argument("--sizes", nargs="+", default=["640x360"])
    parser.add_argument("--qualities", nargs="+", type=int, default=[70, 85])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--deadline-s", type=float, default=.1)
    args = parser.parse_args(argv)
    positive(args.deadline_s, "deadline_s")
    files = sorted(args.frames.glob("*.npy"))
    if len(files) < 200 or args.warmup < 1:
        parser.error("At least 200 replay frames and one warm-up call required")
    if any(not 1 <= q <= 100 for q in args.qualities):
        parser.error("JPEG quality must be 1..100")
    annotations = None
    if args.annotations:
        annotations = json.loads(args.annotations.read_text())
        if not isinstance(annotations, dict) or any(p.name not in annotations or not isinstance(annotations[p.name], list) for p in files):
            parser.error("Annotations must explicitly cover every replay frame, including empty lists")
    sizes = []
    for text in args.sizes:
        try:
            width, height = map(int, text.split("x"))
            if min(width, height) <= 0:
                raise ValueError()
            sizes.append((width, height))
        except ValueError:
            parser.error("Sizes must be positive WIDTHxHEIGHT")
    module, name = args.local_factory.split(":")
    factory = getattr(importlib.import_module(module), name)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, origin=args.origin, model_version=args.model_version,
                    hardware_notes=args.hardware_notes, warmup=args.warmup,
                    deadline_s=args.deadline_s, local_factory=args.local_factory,
                    replay=[dict(file=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
                    modes={}, accuracy={"status": "NOT_MEASURED", "reason": "No annotated truth supplied"})
    if annotations is not None:
        manifest["accuracy"] = dict(status="ANNOTATED", matcher="class-aware confidence-greedy IoU >= 0.5; micro precision/recall",
                                    annotations_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest())
    all_rows = []
    for mode, endpoint in (("local", None), ("LAN", args.lan_endpoint), ("cloud", args.cloud_endpoint)):
        if mode != "local" and not endpoint:
            manifest["modes"][mode] = {"status": "NOT RUN", "reason": "No endpoint supplied"}
            continue
        configurations = []
        for width, height in sizes:
            for quality in ([None] if mode == "local" else args.qualities):
                detector = factory() if mode == "local" else JpegHttpBackend(
                    endpoint, source_id="replay", model_version=args.model_version,
                    timeout_s=args.deadline_s, jpeg_quality=quality)
                rows = []
                try:
                    detector.reset(episode_id="latency-replay")
                    for index in range(args.warmup + len(files)):
                        path = files[index % len(files)] if index < args.warmup else files[index-args.warmup]
                        rgb = np.load(path, allow_pickle=False)
                        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                            raise ValueError("Replay requires uint8 RGB HxWx3")
                        original_height, original_width = rgb.shape[:2]
                        from PIL import Image
                        rgb = np.asarray(Image.fromarray(rgb).resize((width, height)))
                        frame = FrameBundle("latency-replay", index, index*.05, time.monotonic(),
                            EgoPose(VehicleState(0, 0, 0, 0)), rgb,
                            CameraCalibration(width, width, width/2, height/2,
                                              ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))))
                        truth = annotations[path.name] if annotations is not None else None
                        # Validate annotation boxes before inference (not model latency).
                        if truth is not None:
                            match_counts((), truth)
                        batch = None
                        start = time.monotonic()
                        status, error, timing = "OK", "", {}
                        try:
                            batch = detector.detect(frame)
                            if not matches(batch, frame) or batch.metadata.model_version != args.model_version:
                                raise ValueError("Invalid/mismatched batch or model version")
                            timing = getattr(detector, "last_timing", {})
                        except Exception as exc:
                            status, error = "ERROR", f"{type(exc).__name__}: {exc}"
                        total = time.monotonic()-start
                        if index >= args.warmup:
                            counts = (None, None, None) if truth is None else match_counts(
                                batch.detections if status == "OK" else (), truth,
                                scale_x=width/original_width, scale_y=height/original_height)
                            rows.append(dict(mode=mode, width=width, height=height, jpeg_quality=quality,
                                frame=path.name, status=status, error=error, total_s=total,
                                tp=counts[0], fp=counts[1], fn=counts[2],
                                encode_s=timing.get("encode_s"), decode_s=timing.get("decode_s"),
                                http_round_trip_s=timing.get("http_round_trip_s"),
                                server_duration_s=timing.get("server_duration_s"),
                                rtt_residual_s=timing.get("rtt_residual_s")))
                finally:
                    detector.close()
                configurations.append(dict(width=width, height=height, jpeg_quality=quality,
                                           **summarize(rows, args.deadline_s)))
                all_rows.extend(rows)
        manifest["modes"][mode] = dict(status="MEASURED", configurations=configurations)
    with (args.output / "latency.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    lines = ["# Replay comparison", "", f"Origin: {args.origin}", "",
             "Failures are deadline misses; latency quantiles cover successful calls only.", "",
             "| Mode | Size | JPEG | Frames | Errors | p50 ms | p95 ms | Max ms | Miss rate | Precision | Recall |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    def display(value, multiplier=1):
        return "NOT MEASURED" if value is None else f"{value*multiplier:.4f}"
    for mode, evidence in manifest["modes"].items():
        if evidence["status"] == "NOT RUN":
            lines.append(f"| {mode} | NOT RUN | | | | | | | | | |")
        for row in evidence.get("configurations", []):
            fields = [mode, f"{row['width']}x{row['height']}", str(row['jpeg_quality']),
                      str(row['frames']), str(row['failure_count']), display(row['p50_s'],1000),
                      display(row['p95_s'],1000), display(row['max_s'],1000),
                      display(row['deadline_miss_rate']), display(row['accuracy']['precision']),
                      display(row['accuracy']['recall'])]
            lines.append("| " + " | ".join(fields) + " |")
    (args.output / "comparison.md").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    for mode in manifest["modes"]:
        values = [row["total_s"]*1000 for row in all_rows if row["mode"] == mode and row["status"] == "OK"]
        if values:
            axes.plot(sorted(values), np.arange(1, len(values)+1)/len(values), label=mode)
    axes.set(xlabel="Client end-to-end latency (ms)", ylabel="Empirical CDF",
             title=f"{args.origin}: pooled configurations (see CSV)")
    axes.axvline(args.deadline_s*1000, linestyle="--", color="black", label="deadline")
    axes.legend()
    figure.savefig(args.output / "latency.png", bbox_inches="tight")
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
