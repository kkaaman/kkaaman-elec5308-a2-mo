"""RV-06: replay the lane departure monitor on a recorded CARLA run.

Reads ``video.avi`` and ``samples.jsonl`` from a run folder, runs the existing
LaneUNet on every Nth frame, compares the perceived lane offset with the fixed
route at the same look-ahead point, and writes monitor.json, a time-series plot
and HUD screenshots on curves and straights. Diagnostic only; no control.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluation.lane_monitor_replay import replay, report
from evaluation.metrics import load_run
from evaluation.run_logger import file_sha256
from geometry.camera import CameraIntrinsics
from planning.corridor import RouteGeometry
from visualisation.lane_monitor import MIN_CONFIDENCE, LaneDeparture, draw_monitor

ROOT = Path(__file__).resolve().parents[1]
LANE_SHA256 = "10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff"  # RV-02


def calibration_from(manifest):
    recorded = manifest.get("camera_calibration")
    if recorded:
        from autonomy.interfaces import CameraCalibration

        return CameraCalibration(**recorded)
    # Runs recorded before the calibration was logged: the fixed CarlaWorld rig.
    return CameraIntrinsics.from_horizontal_fov(640, 360, math.radians(90)).calibration(
        ((0, 0, 1, 1.5 + 1.38641), (-1, 0, 0, 0), (0, -1, 0, 2.0), (0, 0, 0, 1))
    )


def video_frames(path, counter):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                return
            counter[0] += 1
            yield bgr[:, :, ::-1].copy()
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--mission", type=Path, required=True, help="LiveSetup mission.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "lane_unet_best.pth")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--screenshots", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from perception.lane_inference import LaneSegmenter
    from perception.lane_path import draw_ego_path
    from perception.recorded_detector import verify_local_asset
    from planning.mission import Mission

    verify_local_asset(args.weights, LANE_SHA256)
    manifest, rows, _ = load_run(args.run)
    mission = Mission.load(args.mission)
    if manifest.get("mission_sha256") not in (None, mission.sha256):
        raise ValueError("Run was recorded on a different mission")
    route = RouteGeometry(mission.points_xy_m)
    calibration = calibration_from(manifest)
    counter = [0]
    frames = video_frames(args.run / "video.avi", counter)
    records = replay(
        rows, frames, LaneSegmenter(str(args.weights)), calibration, route, every=args.every
    )
    for _ in frames:  # count any remaining frames for the alignment check
        pass
    args.output.mkdir(parents=True, exist_ok=False)
    result = report(records, rows=len(rows), frames_read=counter[0])
    result.update(
        run=str(args.run),
        run_manifest_sha256=file_sha256(args.run / "manifest.json"),
        mission_sha256=mission.sha256,
        weights_sha256=LANE_SHA256,
        min_confidence=MIN_CONFIDENCE,
    )
    (args.output / "monitor.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    t = [r["sim_time_s"] - records[0]["sim_time_s"] for r in records] if records else []
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    top.plot(t, [r["map_offset_m"] for r in records], color="black", label="Fixed route (map)")
    usable = [(ti, r) for ti, r in zip(t, records) if r["comparison"]["usable"]]
    top.scatter(
        [ti for ti, _ in usable],
        [r["departure"]["signed_offset_m"] for _, r in usable],
        s=8,
        color="tab:orange",
        label=f"Lane monitor (confidence >= {MIN_CONFIDENCE})",
    )
    for ti, r in zip(t, records):
        if r["curve"]:
            top.axvspan(ti - 0.05, ti + 0.05, color="0.9", linewidth=0)
    top.set(
        ylabel="Look-ahead offset (m, +left)",
        title="RV-06 lane monitor vs fixed route (grey: curve)",
    )
    top.legend(fontsize=8)
    bottom.plot(t, [r["departure"]["confidence"] for r in records], color="tab:blue")
    bottom.axhline(MIN_CONFIDENCE, color="red", linestyle="--", linewidth=1)
    bottom.set(xlabel="Simulation time (s)", ylabel="Confidence")
    fig.tight_layout()
    fig.savefig(args.output / "monitor.png", dpi=130)
    plt.close(fig)

    shots = args.output / "screenshots"
    shots.mkdir()
    curve = [r for r in records if r["curve"]]
    straight = [r for r in records if not r["curve"]]
    wanted = set()
    for group in (curve, straight):
        if group:
            step = max(1, len(group) // max(1, args.screenshots // 2))
            wanted.update(r["index"] for r in group[::step][: args.screenshots // 2])
    by_index = {r["index"]: r for r in records}
    for index, rgb in enumerate(video_frames(args.run / "video.avi", [0])):
        if index not in wanted:
            continue
        r = by_index[index]
        image = draw_ego_path(rgb, r["path"]) if r["path"] is not None else rgb
        image = draw_monitor(image, LaneDeparture(**r["departure"]), r["map_offset_m"])
        label = "curve" if r["curve"] else "straight"
        cv2.imwrite(str(shots / f"{label}_{r['frame_id']}.png"), image[:, :, ::-1])
    summary = result["summary"]
    print(
        f"{result['status']} | usable {summary['all']['usable_frames']}/{len(records)} |"
        f" curve sign agreement {summary['curve']['sign_agreement']} |"
        f" median |diff| {summary['all']['median_abs_difference_m']} | {args.output}"
    )


if __name__ == "__main__":
    main()
