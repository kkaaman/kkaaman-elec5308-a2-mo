"""Depth errors by range and paired learned-mask ablation on recorded RGB-D."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import local_file, read_dataset, save_json, sha256
from evaluation.perception_replay import axial_image
from geometry.camera import CameraIntrinsics
from obstacle_perception.box_depth import estimate_box_depth
from obstacle_perception.types import PixelBox


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "depth-calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--segmentation-weights", type=Path)
    args = parser.parse_args()
    manifest, rows = read_dataset(args.dataset)
    depth = json.loads(args.depth_calibration.read_text())
    if depth["status"] != "MEASURED_PASS" or depth.get("opendrive_sha256") != manifest.get(
        "opendrive_sha256"
    ):
        raise ValueError("Matching map depth calibration required")
    rows = [r for r in rows if r["split"] == args.split and r.get("sequence_index") == 0]
    args.output.mkdir(parents=True, exist_ok=False)
    width, height = manifest["image_size_wh"]
    rig = manifest["camera"]
    cal = CameraIntrinsics.from_horizontal_fov(
        width, height, math.radians(rig["fov_deg"])
    ).calibration(
        (
            (0, 0, 1, rig["x_actor_m"] + 1.38641),
            (-1, 0, 0, 0),
            (0, -1, 0, rig["z_actor_m"]),
            (0, 0, 0, 1),
        )
    )
    segmenter = None
    if args.segmentation_weights:
        from perception.recorded_detector import verify_local_asset
        from segmentation.segmentation import SemanticSegmenter

        verify_local_asset(
            args.segmentation_weights,
            "a631440d3fe2ccd2554631bb49b1352c0efbbaffc2316caa79c288757e8dde2a",
        )
        segmenter = SemanticSegmenter(str(args.segmentation_weights))
    references = []
    pairs = []
    timings = []
    for row in rows:
        for sample in row.get("surface_references", []):
            ref = sample["reference"]
            value = sample["raw_depth_m"]
            if ref and value is not None:
                references.append(
                    dict(
                        sample_id=row["sample_id"],
                        class_name=sample["class_name"],
                        distance_m=ref[depth["convention"] + "_m"],
                        error_m=value - ref[depth["convention"] + "_m"],
                    )
                )
        raw = np.load(local_file(args.dataset, row["depth"]), allow_pickle=False)
        axial = axial_image(raw, cal, depth["convention"])
        with Image.open(local_file(args.dataset, row["instance"])) as image:
            instance = np.asarray(image)
        ids = instance[:, :, 1].astype(np.uint32) * 256 + instance[:, :, 2]
        mask = None
        if segmenter:
            with Image.open(local_file(args.dataset, row["rgb"])) as image:
                rgb = np.asarray(image.convert("RGB"))
            started = time.perf_counter()
            mask = segmenter.predict(rgb)
            timings.append(time.perf_counter() - started)
        for truth in row["visible_boxes"]:
            name = truth["class_name"]
            tag = 12 if name == "person" else 14
            gt = axial[
                (instance[:, :, 0] == tag) & (ids == truth["instance_code"]) & np.isfinite(axial)
            ]
            if len(gt) < 9:
                continue
            reference = float(np.median(gt))
            box = PixelBox(*truth["xyxy"])
            plain = estimate_box_depth(axial, box, cal)
            masked = None
            if mask is not None:
                candidate = axial.copy()
                candidate[mask != (11 if name == "person" else 13)] = np.nan
                masked = estimate_box_depth(candidate, box, cal)
            pairs.append(
                dict(
                    sample_id=row["sample_id"],
                    class_name=name,
                    distance_m=reference,
                    plain_error_m=plain.axial_depth_m - reference if plain.valid else None,
                    masked_error_m=masked.axial_depth_m - reference
                    if masked and masked.valid
                    else None,
                )
            )
    bins = []
    for name in ("car", "person"):
        for centre in (5, 10, 15, 20, 25, 30):
            selected = [
                r
                for r in references
                if r["class_name"] == name and abs(r["distance_m"] - centre) <= 2.5
            ]
            baseline = [
                r for r in pairs if r["class_name"] == name and abs(r["distance_m"] - centre) <= 2.5
            ]
            bins.append(
                dict(
                    class_name=name,
                    range_centre_m=centre,
                    geometry_ray_samples=len(selected),
                    independent_scenes=len({r["sample_id"] for r in selected}),
                    median_abs_geometry_error_m=float(
                        np.median([abs(r["error_m"]) for r in selected])
                    )
                    if selected
                    else None,
                    box_count=len(baseline),
                    invalid_depth_rate=sum(r["plain_error_m"] is None for r in baseline)
                    / len(baseline)
                    if baseline
                    else None,
                )
            )
    paired = [
        r for r in pairs if r["plain_error_m"] is not None and r["masked_error_m"] is not None
    ]
    improvement = (
        float(np.median([abs(r["plain_error_m"]) - abs(r["masked_error_m"]) for r in paired]))
        if paired
        else None
    )
    report = dict(
        status="MEASURED_REVIEW_REQUIRED",
        split=args.split,
        depth_convention=depth["convention"],
        bins=bins,
        dataset_index_sha256=manifest["samples_sha256"],
        depth_calibration_sha256=sha256(args.depth_calibration),
        segmentation_sha256=sha256(args.segmentation_weights) if segmenter else None,
        masking_enabled_in_runtime=False,
        paired_valid_boxes=len(paired),
        median_mask_improvement_m=improvement,
        mask_first_s=timings[0] if timings else None,
        mask_warm_p95_s=float(np.percentile(timings[1:], 95)) if len(timings) > 1 else None,
        plain_invalid=sum(r["plain_error_m"] is None for r in pairs),
        masked_invalid=sum(r["masked_error_m"] is None for r in pairs) if segmenter else None,
        scope="Ray geometry checks raw-depth convention/range; oracle visible boxes isolate masking effect. No driving false-stop/FPS benefit claimed; mask stays OFF pending end-to-end evidence.",
        pairs=pairs,
        references=references,
    )
    save_json(args.output / "depth_analysis.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ("car", "person"):
        xs = [
            r
            for r in bins
            if r["class_name"] == name and r["median_abs_geometry_error_m"] is not None
        ]
        ax.plot(
            [r["range_centre_m"] for r in xs],
            [r["median_abs_geometry_error_m"] for r in xs],
            "o-",
            label=name,
        )
    ax.set(
        xlabel="Range bin centre (m)",
        ylabel="Median absolute ray error (m)",
        title="Measured raw-depth geometry agreement; missing bins remain missing",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "depth_ranges.png")
    plt.close(fig)
    print(f"{len(references)} geometry samples; {len(pairs)} box comparisons | {args.output}")


if __name__ == "__main__":
    main()
