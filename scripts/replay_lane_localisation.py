"""Offline lane-aided localisation on recorded runs (GNSS-only vs camera-lane-aided).

For each run recorded with ``--localisation``:
1. LaneUNet runs on the run's ``video.avi`` (the RV-06 lane monitor, every Nth frame)
   to get the camera lane offset 8 m ahead of the rear axle.
2. Three estimators are replayed on the run's logged raw inputs: GNSS-only (as run
   live; it must reproduce the logged estimate), GNSS + lane, and GNSS + lane with the
   GNSS offset and wheel-speed scale as states.
3. Each estimate is scored against the logged true pose.

Nothing is driven: the car drove on whatever the run used.

Usage (repository root, project venv):
    python scripts/replay_lane_localisation.py <stage or run folder> [...] \
        --mission <mission.json> --output <new folder>

Writes report.json and lane_localisation.png into the output folder.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from replay_lane_monitor import LANE_SHA256, calibration_from, video_frames  # noqa: E402

from autonomy.lane_localisation import LaneAiding  # noqa: E402
from autonomy.localisation import LocalisationConfig  # noqa: E402
from evaluation.lane_localisation_replay import (  # noqa: E402
    VARIANTS,
    lane_measurements,
    lane_quality,
    replay,
    summarise,
)
from evaluation.lane_monitor_replay import replay as lane_monitor  # noqa: E402
from evaluation.metrics import load_run  # noqa: E402
from evaluation.run_logger import file_sha256  # noqa: E402
from planning.corridor import RouteGeometry  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LABELS = dict(
    gnss="GNSS only (as run live)",
    gnss_lane="GNSS + camera lane",
    gnss_lane_offset="GNSS + camera lane, offset and wheel scale estimated",
)
COLOURS = dict(gnss="#2a78d6", gnss_lane="#eb6834", gnss_lane_offset="#1baf7a")
INK, SECOND, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def run_folders(paths):
    found = []
    for path in paths:
        path = Path(path)
        if (path / "samples.jsonl").exists():
            found.append(path)
        else:
            parent = path / "runs" if (path / "runs").is_dir() else path
            found.extend(sorted(p.parent for p in parent.glob("*/samples.jsonl")))
    if not found:
        raise SystemExit(f"No run folders under {', '.join(map(str, paths))}")
    return found


def evaluate_run(folder, manifest, rows, route, aiding, records):
    """Replay the three estimators on one run, given its lane-monitor records."""
    recorded = manifest["localisation"]
    config = LocalisationConfig.from_dict(recorded["config"])
    lanes = lane_measurements(records, aiding.min_confidence)
    drive_mode = manifest.get("pose_source") == "estimate"
    trajectory, info = replay(rows, route, config, aiding, lanes, drive_mode=drive_mode)
    sensors = recorded.get("sensors", {})
    folder = Path(folder)
    name = (
        f"{folder.parent.parent.name}/{folder.name}"
        if folder.parent.name == "runs"
        else folder.name
    )
    result = dict(
        run=name,
        seed=manifest.get("seed"),
        mode=recorded.get("mode"),
        run_manifest_sha256=file_sha256(folder / "manifest.json"),
        true_gnss_offset_m=sensors.get("gnss_bias_m"),
        true_wheel_scale=sensors.get("wheel_scale"),
        lane_monitor=lane_quality(records, aiding.min_confidence),
        logged_rows=len(rows),
        **info,
        variants={variant: summarise(trajectory[variant]) for variant in VARIANTS},
    )
    return result, trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="stage folders or run folders")
    parser.add_argument("--mission", type=Path, required=True, help="the runs' mission.json")
    parser.add_argument(
        "--lane-config", type=Path, default=ROOT / "configs/lane_localisation_consumer.json"
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "lane_unet_best.pth")
    parser.add_argument("--every", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from perception.lane_inference import LaneSegmenter
    from perception.recorded_detector import verify_local_asset
    from planning.mission import Mission

    verify_local_asset(args.weights, LANE_SHA256)
    mission = Mission.load(args.mission)
    route = RouteGeometry(mission.points_xy_m)
    aiding = LaneAiding.load(args.lane_config)
    segmenter = LaneSegmenter(str(args.weights))
    args.output.mkdir(parents=True, exist_ok=False)

    results, trajectories = [], {}
    for folder in run_folders(args.runs):
        manifest, rows, _ = load_run(folder)
        if not manifest.get("localisation"):
            print(f"skip {folder}: recorded without --localisation")
            continue
        if manifest.get("mission_sha256") not in (None, mission.sha256):
            raise ValueError(f"{folder} was recorded on a different mission")
        counter = [0]
        records = lane_monitor(
            rows,
            video_frames(folder / "video.avi", counter),
            segmenter,
            calibration_from(manifest),
            route,
            every=args.every,
        )
        result, trajectory = evaluate_run(folder, manifest, rows, route, aiding, records)
        result["video_frames"] = counter[0]
        trajectories[result["run"]] = trajectory
        results.append(result)
        print_run(result)
    if not results:
        raise SystemExit("No runs with localisation records")

    report = dict(
        status="OFFLINE_REPLAY_MEASURED_REVIEW_REQUIRED",
        scope=(
            "Offline replay on recorded runs: the car drove on the pose its run used; only"
            " the estimates change. Lane offsets come from LaneUNet on the recorded video."
        ),
        mission_sha256=mission.sha256,
        lane_weights_sha256=LANE_SHA256,
        lane_config=aiding.as_dict(),
        lane_config_sha256=file_sha256(args.lane_config),
        every=args.every,
        runs=results,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    plot(trajectories, args.output / "lane_localisation.png")
    print(f"{report['status']} | {len(results)} runs | {args.output}")


def print_run(result):
    print(
        f"{result['run']} seed {result['seed']} | replay vs logged "
        f"{result['replay_max_difference']:.2e} | lane usable "
        f"{result['lane_monitor'].get('usable')}/{result['lane_monitor']['frames']}, median "
        f"|lane-map| {result['lane_monitor'].get('median_abs_m')}"
    )
    for name in VARIANTS:
        s = result["variants"][name]
        if s is None:
            print(f"  {name:17s} no moving samples")
            continue
        approach = (s["goal_approach_along_error_m"] or {}).get("mean", float("nan"))
        print(
            f"  {name:17s} lateral mean {s['lateral_error_m']['mean']:.3f} p95 "
            f"{s['lateral_error_m']['p95']:.3f} max {s['lateral_error_m']['max']:.3f} m | "
            f"along p95 {s['along_error_m']['p95']:.3f} m, last 20 m mean "
            f"{approach:.3f} m | yaw p95 {s['yaw_error_deg']['p95']:.2f} deg"
        )
    estimated = result["final_gnss_offset_m"]["gnss_lane_offset"]
    print(
        f"  GNSS offset true {result['true_gnss_offset_m']} estimated {[round(v, 3) for v in estimated]}"
    )


def plot(trajectories, path):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": SECOND,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, label in zip(
        axes,
        ("lateral_error_m", "along_error_m"),
        ("|Lateral error| (m, log)", "|Along-route error| (m, log)"),
    ):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.8, which="major")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for name in VARIANTS:
            for i, trajectory in enumerate(trajectories.values()):
                s = [r["progress_m"] for r in trajectory[name]]
                e = [max(1e-3, abs(r[key])) for r in trajectory[name]]
                ax.plot(
                    s,
                    e,
                    color=COLOURS[name],
                    lw=1.2,
                    label=LABELS[name] if i == 0 else None,
                )
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 5)
        ax.set_ylabel(label)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].set_xlabel("Distance along the route (m)")
    fig.suptitle(
        f"Pose error by estimator, offline replay on {len(trajectories)} recorded runs",
        x=0.02,
        ha="left",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    main()
