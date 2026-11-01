"""Replay RV-06 monitor on the prescribed A1 display recording and matched poses.

Uses recorded camera/vehicle geometry; display frames are not pristine RGB or
exact sensor-frame timestamps. This is a convention/overlay check, not accuracy.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

from autonomy.paths import model_path
from evaluation.a1_comparison import LaneCentrelines
from geometry.camera import CameraIntrinsics
from perception.lane_inference import LaneSegmenter
from perception.lane_path import extract_ego_path
from planning.lane_graph import LaneTopology
from visualisation.hud import draw_departure_overlay
from visualisation.lane_departure import LaneDepartureMonitor, lane_estimate_from_centerline


def replay(clip, output, topology_path):
    clip, output = Path(clip), Path(output)
    manifest = json.loads((clip / "manifest.json").read_text())
    camera = next(a for a in manifest["spawned_actors"] if a["blueprint"] == "sensor.camera.rgb")
    mount = camera["requested_pose_carla"]
    if any(abs(mount[k]) > 1e-5 for k in ("pitch_deg", "yaw_deg", "roll_deg")):
        raise ValueError("This recorded projection needs a zero-rotation optical mount")
    vehicle = manifest["actual_vehicle"]
    pose = vehicle["first_recorded_pose"]
    rear = [w["position"] for w in vehicle["wheels"] if w["max_steer_angle_deg"] == 0]
    if len(rear) != 2:
        raise ValueError("Two non-steering rear wheel positions required")
    rear_world = np.mean(rear, axis=0) / 100
    yaw = math.radians(pose["yaw_deg"])
    offset = -float(
        (rear_world[:2] - [pose["x"], pose["y"]]) @ np.array([math.cos(yaw), math.sin(yaw)])
    )
    if not 0 < offset < 3:
        raise ValueError("Invalid measured actor-to-rear-axle offset")
    samples = list(csv.DictReader((clip / "samples.csv").open()))
    index = list(csv.DictReader((clip / "video_frames.csv").open()))
    video = cv2.VideoCapture(str(clip / "video.avi"))
    ok, first = video.read()
    if not ok:
        raise ValueError("Recording has no readable video frame")
    height, width = first.shape[:2]
    # The recorder's video may be a resized display; use its actual dimensions.
    fov = math.radians(float(camera["attributes"]["fov"]))
    matrix = (
        (0, 0, 1, mount["x"] + offset),
        (-1, 0, 0, -mount["y"]),
        (0, -1, 0, mount["z"]),
        (0, 0, 0, 1),
    )
    cal = CameraIntrinsics.from_horizontal_fov(width, height, fov).calibration(matrix)
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lanes = LaneCentrelines(LaneTopology.load(topology_path))
    segmenter, monitor = LaneSegmenter(model_path("lane")), LaneDepartureMonitor()
    rows = []
    previous = None
    output.mkdir(parents=True, exist_ok=False)
    try:
        for record in index:
            ok, bgr = video.read()
            if not ok:
                raise ValueError("Video shorter than recorded frame index")
            sample = samples[int(record["latest_control_sample"])]
            x, y, angle = (float(sample[k]) for k in ("actor_x_m", "actor_y_m", "actor_yaw_rad"))
            x -= offset * math.cos(angle)
            y -= offset * math.sin(angle)
            map_cte, section, junction = lanes.signed_offset(x, y)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            path = extract_ego_path(segmenter.predict(rgb))
            t, frame = float(record["sim_time_s"]), int(record["video_frame"])
            estimate = (
                None
                if path is None
                else lane_estimate_from_centerline(
                    path["centerline"], cal, frame_id=frame, sim_time_s=t
                )
            )
            report = monitor.update(estimate, None if junction else map_cte, now_sim_time_s=t)
            turn = "unknown"
            if previous is not None and t > previous[0]:
                rate = math.atan2(math.sin(angle - previous[1]), math.cos(angle - previous[1])) / (
                    t - previous[0]
                )
                turn = "left" if rate > 0.01 else "right" if rate < -0.01 else "straight"
            previous = (t, angle)
            rows.append(
                dict(
                    report.as_record(),
                    display_frame=frame,
                    reference_rear_axle_xy_m=[x, y],
                    map_section=section,
                    in_junction=bool(junction),
                    turn=turn,
                    actor_reference_cte_m=float(sample["nearest_lane_cte_signed_m"]),
                )
            )
            if frame % 5 == 0:
                image = draw_departure_overlay(rgb, report)
                if not cv2.imwrite(
                    str(output / f"frame_{frame:06d}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                ):
                    raise OSError("Failed to save recorded overlay")
    finally:
        video.release()
    valid = [
        r
        for r in rows
        if r["lane_monitor_valid"] and not r["in_junction"] and abs(r["lane_map_cte_m"]) > 0.1
    ]
    counts = {}
    for turn in ("left", "right", "straight"):
        chosen = [r for r in valid if r["turn"] == turn]
        agree = sum(
            math.copysign(1, r["lane_image_offset_m"]) == math.copysign(1, r["lane_map_cte_m"])
            for r in chosen
        )
        counts[turn] = dict(
            valid_samples=len(chosen),
            sign_agreement=agree,
            mean_abs_difference_m=float(np.mean([abs(r["lane_difference_m"]) for r in chosen]))
            if chosen
            else None,
        )
    summary = dict(
        frames=len(rows),
        valid_nonjunction_samples=len(valid),
        turn_results=counts,
        measured_rear_axle_offset_m=offset,
        camera_mount=mount,
        display_dimensions=[width, height],
        video_sha256=hashlib.sha256((clip / "video.avi").read_bytes()).hexdigest(),
        scope="Processed display frames with approximate matched control pose; not pristine sensor accuracy or exact frame-aligned latency",
        operational_acceptance=False,
    )
    (output / "monitor.json").write_text(
        json.dumps(dict(summary=summary, frames=rows), indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--topology", type=Path, default=Path("tests/fixtures/Town10HD.lane_graph.json")
    )
    args = parser.parse_args()
    print(json.dumps(replay(args.clip, args.output, args.topology), indent=2))
