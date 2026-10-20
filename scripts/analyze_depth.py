"""Depth errors by range and paired learned-mask ablation on recorded RGB-D."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import local_file, read_dataset, save_json, sha256
from evaluation.masking import masking_decision
from geometry.camera import CameraIntrinsics
from obstacle_perception.box_depth import estimate_box_depth
from obstacle_perception.types import PixelBox


def axial_image(raw, calibration, convention):
    if convention not in {"axial", "radial"}:
        raise ValueError("Measured depth convention required")
    if convention == "axial":
        return raw.copy()

    vv, uu = np.indices(raw.shape)
    scale = np.sqrt(
        1
        + ((uu - calibration.cx_px) / calibration.fx_px) ** 2
        + ((vv - calibration.cy_px) / calibration.fy_px) ** 2
    )
    return (raw / scale).astype(np.float32)


def overlap(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (
        (a[2] - a[0]) * (a[3] - a[1])
        + (b[2] - b[0]) * (b[3] - b[1])
        - area
    )
    return area / union if union > 0 else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "depth-calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--segmentation-weights", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        help="Existing YOLO weights: also ablate masking on detector boxes (BL-B-05 decision)",
    )
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
    detector = None
    if args.weights:
        from perception import ObjectDetector, PerceptionConfig
        from perception.recorded_detector import verify_local_asset

        asset = json.loads(
            (Path(__file__).resolve().parents[1] / "configs/models.yaml").read_text()
        )["models"][0]
        verify_local_asset(args.weights, asset["sha256"])
        detector = ObjectDetector(PerceptionConfig(model_name=str(args.weights.resolve())))
    references = []
    pairs = []
    detector_pairs = []
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
        mask = None
        rgb = None
        if segmenter or detector:
            with Image.open(local_file(args.dataset, row["rgb"])) as image:
                rgb = np.asarray(image.convert("RGB")).copy()
        detections = detector.detect(rgb) if detector else ()
        if segmenter:
            started = time.perf_counter()
            mask = segmenter.predict(rgb)
            timings.append(time.perf_counter() - started)
        target = row.get("target_evaluation_only", {})
        if not target.get("spawned"):
            continue

        target_instance_code = int(target["actor_id"]) * 256
        target_boxes = [
            truth
            for truth in row["visible_boxes"]
            if truth["instance_code"] == target_instance_code
        ]
        if len(target_boxes) != 1:
            continue

        truth = target_boxes[0]
        name = truth["class_name"]

        target_refs = [
            sample["reference"]
            for sample in row.get("surface_references", [])
            if sample.get("reference") is not None
            and sample["class_name"] == name
        ]
        if not target_refs:
            continue

        reference = float(
            np.median(
                [
                    ref[depth["convention"] + "_m"]
                    for ref in target_refs
                ]
            )
        )

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
                plain_error_m=plain.axial_depth_m - reference
                if plain.valid
                else None,
                masked_error_m=masked.axial_depth_m - reference
                if masked and masked.valid
                else None,
            )
        )

        if detector:
            best, score = None, 0.5
            for detection in detections:
                b = detection.bounding_box
                candidate = overlap(
                    (b.x_min, b.y_min, b.x_max, b.y_max),
                    truth["xyxy"],
                )
                if detection.class_name == name and candidate >= score:
                    best, score = detection, candidate

            if best is not None:
                b = best.bounding_box
                det_box = PixelBox(
                    b.x_min,
                    b.y_min,
                    b.x_max,
                    b.y_max,
                )
                plain_det = estimate_box_depth(axial, det_box, cal)

                masked_det = None
                if mask is not None:
                    candidate = axial.copy()
                    candidate[
                        mask != (11 if name == "person" else 13)
                    ] = np.nan
                    masked_det = estimate_box_depth(
                        candidate,
                        det_box,
                        cal,
                    )

                detector_pairs.append(
                    dict(
                        sample_id=row["sample_id"],
                        class_name=name,
                        distance_m=reference,
                        iou=score,
                        plain_error_m=(
                            plain_det.axial_depth_m - reference
                            if plain_det.valid
                            else None
                        ),
                        masked_error_m=(
                            masked_det.axial_depth_m - reference
                            if masked_det and masked_det.valid
                            else None
                        ),
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
                    geometry_samples=len(selected),
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
    mask_p95 = float(np.percentile(timings[1:], 95)) if len(timings) > 1 else None
    decision = masking_decision(pairs, detector_pairs, mask_p95)
    report = dict(
        status="MEASURED_REVIEW_REQUIRED",
        masking_decision=decision,
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
        scope="Evaluation-only target geometry checks raw-depth range accuracy; oracle target boxes isolate box-depth behaviour. No driving false-stop/FPS benefit claimed; mask stays OFF pending end-to-end evidence.",
        pairs=pairs,
        detector_pairs=detector_pairs,
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
        ylabel="Median absolute geometry error (m)",
        title="Measured raw-depth target-geometry agreement; missing bins remain missing",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output / "depth_ranges.png")
    plt.close(fig)
    print(
        f"{len(references)} geometry samples; {len(pairs)} oracle / {len(detector_pairs)}"
        f" detector box comparisons; masking: {decision['decision']} | {args.output}"
    )


if __name__ == "__main__":
    main()
