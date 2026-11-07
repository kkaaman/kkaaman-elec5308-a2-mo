"""Replay a LiDAR sweep set through ground removal, clustering, tracking and fusion.

Writes a JSON summary (per-frame governor inputs plus aggregates). The default
input is the checked-in synthetic fixture; pass a live capture's sweeps.json from
scripts/capture_lidar_sweeps.py to replay recorded data. ``--regenerate-fixture``
rewrites the synthetic fixture deterministically.
"""

import argparse
import json
from pathlib import Path

from autonomy.lidar_replay import replay_path
from autonomy.paths import FIXTURES_DIR
from obstacle_perception.lidar_synthetic import build_fixture

DEFAULT_INPUT = FIXTURES_DIR / "lidar" / "sweeps.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="sweeps.json manifest")
    parser.add_argument("--output", type=Path, default=None, help="summary JSON path")
    parser.add_argument(
        "--half-width", type=float, default=None, help="corridor half-width override (m)"
    )
    parser.add_argument(
        "--ahead", type=float, default=None, help="corridor look-ahead override (m)"
    )
    parser.add_argument(
        "--regenerate-fixture",
        action="store_true",
        help="rewrite the synthetic fixture before replay",
    )
    args = parser.parse_args()
    if args.regenerate_fixture:
        build_fixture(DEFAULT_INPUT.parent)
    result = replay_path(args.input, half_width_m=args.half_width, ahead_m=args.ahead)
    summary = result["summary"]
    print(
        f"{summary['frames']} frames: {summary['sweep_valid_frames']} valid sweeps, "
        f"{summary['fused_valid_frames']} fused valid, {summary['hold_frames']} HOLD; "
        f"max |LiDAR gap error| = {summary['max_abs_lidar_gap_error_m']} m; "
        f"depth-loss frames treated clear = {summary['depth_loss_frames_treated_clear']}"
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
