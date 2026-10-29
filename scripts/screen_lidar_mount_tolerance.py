"""Compact orientation-tolerance follow-up; retain all source screening output."""

import argparse
from pathlib import Path

from scripts import screen_lidar_geometry as screening

screening.CANDIDATES = [
    (
        f"front_z{height:g}_pitch{pitch:+g}",
        dict(
            x_actor_m=2.5,
            z_actor_m=height,
            pitch_deg=pitch,
            lower_fov_deg=-85,
            channels=128,
            points_per_second=1280000,
        ),
    )
    for height in (0.9, 1.3, 1.5)
    for pitch in (-1.0, 0.0, 1.0)
]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="172.28.176.1")
    parser.add_argument("--site", type=int, default=22)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--output", type=Path, required=True)
    screening.screen(parser.parse_args())
