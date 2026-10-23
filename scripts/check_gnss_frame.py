"""Which way is north? Fit the GNSS fixes in a capture run against the truth ego position.

No CARLA needed: ``samples.jsonl`` holds each sample's GNSS fix and the truth rear-axle
position in the core frame (x forward-east of CARLA, y = -CARLA y). Prints the metres of
latitude and longitude per metre of core x and y, so a mirrored axis shows as a negative
diagonal. Live G1b curve run (23 Oct) ended with the route pose's y opposite to the
odometry's; this says whether the map's geolocation or the stack is to blame.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6371000.0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", type=Path, required=True, action="append")
    args = parser.parse_args(argv)
    for run in args.run:
        xy, ll = [], []
        for line in (run / "samples.jsonl").read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            s = json.loads(line)
            if s.get("gnss") is None:
                continue
            e = s["truth"]["ego_rear_axle"]
            xy.append((e["x_m"], e["y_m"]))
            ll.append((s["gnss"]["latitude_deg"], s["gnss"]["longitude_deg"]))
        xy, ll = np.array(xy), np.array(ll)
        lat0 = ll[:, 0].mean()
        north = (ll[:, 0] - lat0) * math.radians(1.0) * EARTH_RADIUS_M
        east = (ll[:, 1] - ll[:, 1].mean()) * math.radians(1.0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        A = np.column_stack((xy - xy.mean(axis=0), np.ones(len(xy))))
        cn, *_ = np.linalg.lstsq(A, north, rcond=None)
        ce, *_ = np.linalg.lstsq(A, east, rcond=None)
        print(json.dumps(dict(
            run=str(run), samples=int(len(xy)), reference_lat=round(float(lat0), 6),
            north_per_core_x=round(float(cn[0]), 4), north_per_core_y=round(float(cn[1]), 4),
            east_per_core_x=round(float(ce[0]), 4), east_per_core_y=round(float(ce[1]), 4),
            verdict="north = +core y (right-handed)" if cn[1] > 0.9 else "north = -core y (MIRRORED)" if cn[1] < -0.9 else "unclear",
        ), indent=2))


if __name__ == "__main__":
    main()
