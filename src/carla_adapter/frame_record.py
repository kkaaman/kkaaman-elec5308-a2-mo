"""Record the sensor frames a mapless drive saw, so the driver can be replayed offline.

Three live turn runs on Town10HD (24 Oct) placed the exit lane 1.1 m off, from a kerb
read at 27 m; the console and the drive log showed the symptom, not the cause. A replay
(``scripts/replay_mapless_frames.py``) runs the same driver code over the recorded frames
on the development machine, with the terrain grid, exit-line stations and truth to hand.

What is written under ``<run>/frames``:
- ``meta.json``: episode and calibration ids, the LiDAR's name and extrinsics, the suite
  (name, sha and the full specification, so the recording replays without the config file).
- ``frames.jsonl``: one row per tick: frame id, times, speed, GNSS fix, IMU sample, the
  per-sensor status list and the names of the radars stored that tick (cameras are not
  stored; the stack's semantic layer is replayed off).
- ``t<tick>.npz``: the LiDAR sweep as int16 centimetres (x, y, z; intensity is range-only
  in the simulator and unused), ~90 KB a tick against 250 KB as float32, and each radar's
  returns as float32 (range, azimuth, elevation, range rate; ~300 a tick, 5 KB) under
  ``radar_<name>``. Schema 1 recordings (before 29 Oct) hold no radar: the PC 1 run in
  which the oncoming car was first tracked 26 m away could not be replayed with it.

The replay is open loop: the car follows the logged motion whatever the replayed driver
commands, so it answers questions about perception and estimation, not control.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mapless.frame import (
    GnssFix,
    ImuSample,
    LidarSweep,
    RadarReturns,
    SensorFrame,
    SensorState,
    SensorStatus,
)


class FrameRecorder:
    def __init__(self, directory, suite, calibration_id, episode_id, extrinsics):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lidar = suite.lidars[0].name
        (self.directory / "meta.json").write_text(
            json.dumps(
                dict(
                    schema="group7-mapless-frames-2",
                    episode_id=episode_id,
                    calibration_id=calibration_id,
                    lidar=self.lidar,
                    lidar_to_ego=np.asarray(extrinsics[self.lidar]).tolist(),
                    radar_to_ego={s.name: np.asarray(extrinsics[s.name]).tolist() for s in suite.radars if s.name in extrinsics},
                    suite=suite.name,
                    suite_sha256=suite.sha256,
                    suite_spec=suite.as_dict(),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        self._rows = open(self.directory / "frames.jsonl", "w", encoding="utf-8")
        self.ticks = 0
        reach = max(float(s.range_m) for s in suite.lidars) + float(np.abs(np.asarray(extrinsics[self.lidar])[:3, 3]).max())
        if reach * 100.0 >= np.iinfo(np.int16).max:
            raise ValueError(f"LiDAR reach {reach:.0f} m does not fit int16 centimetres (327 m)")

    def record(self, tick, frame):
        row = dict(
            tick=tick,
            frame_id=frame.frame_id,
            sim_time_s=frame.sim_time_s,
            received_monotonic_s=frame.received_monotonic_s,
            speed_mps=frame.speed_mps,
            gnss=None if frame.gnss is None else [frame.gnss.latitude_deg, frame.gnss.longitude_deg, frame.gnss.altitude_m],
            imu=None if frame.imu is None else dict(accel_mps2=list(frame.imu.accel_mps2), gyro_rps=list(frame.imu.gyro_rps), compass_rad=frame.imu.compass_rad),
            status=[dict(name=s.name, kind=s.kind, state=s.state.value, capture_time_s=s.capture_time_s, detail=s.detail) for s in frame.status],
            lidar=any(s.name == self.lidar for s in frame.lidars),
            radars=[r.name for r in frame.radars],
        )
        self._rows.write(json.dumps(row) + "\n")
        self._rows.flush()  # the recording exists to debug runs that die
        arrays = {f"radar_{r.name}": np.asarray(r.returns, dtype=np.float32) for r in frame.radars}
        for sweep in frame.lidars:
            if sweep.name == self.lidar:
                arrays["xyz_cm"] = np.round(sweep.points[:, :3] * 100.0).astype(np.int16)
        if arrays:
            np.savez_compressed(self.directory / f"t{tick:05d}.npz", **arrays)
        self.ticks += 1

    def close(self):
        self._rows.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def iter_frames(directory):
    """Yield (tick, SensorFrame) from a recording, in order."""
    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    extrinsics = np.asarray(meta["lidar_to_ego"], dtype=float)
    radar_to_ego = meta.get("radar_to_ego", {})  # absent in schema 1: no radar was stored
    with open(directory / "frames.jsonl", encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            lidars, radars = (), ()
            stored = [n for n in row.get("radars", ()) if n in radar_to_ego]
            arrays = np.load(directory / f"t{row['tick']:05d}.npz") if row["lidar"] or stored else None
            if row["lidar"]:
                xyz = arrays["xyz_cm"].astype(np.float32) / 100.0
                lidars = (LidarSweep(meta["lidar"], np.column_stack((xyz, np.zeros(len(xyz), dtype=np.float32))), extrinsics),)
            radars = tuple(RadarReturns(n, arrays[f"radar_{n}"], radar_to_ego[n]) for n in stored)
            # Cameras are not stored, nor radars before schema 2, so their statuses are
            # dropped (a frame may carry OK status only for data it holds).
            status_rows = [
                s
                for s in row["status"]
                if s["kind"] != "camera" and not (s["kind"] == "lidar" and s["name"] != meta["lidar"]) and not (s["kind"] == "radar" and s["name"] not in radar_to_ego)
            ]
            status = tuple(SensorStatus(s["name"], s["kind"], SensorState(s["state"]), s["capture_time_s"], s.get("detail", "")) for s in status_rows)
            frame = SensorFrame(
                episode_id=meta["episode_id"],
                frame_id=int(row["frame_id"]),
                sim_time_s=float(row["sim_time_s"]),
                received_monotonic_s=float(row["received_monotonic_s"]),
                cameras=(),
                lidars=lidars,
                radars=radars,
                gnss=None if row["gnss"] is None else GnssFix(*row["gnss"]),
                imu=None if row["imu"] is None else ImuSample(row["imu"]["accel_mps2"], row["imu"]["gyro_rps"], row["imu"]["compass_rad"]),
                speed_mps=float(row["speed_mps"]),
                status=status,
                calibration_id=meta["calibration_id"],
            )
            yield int(row["tick"]), frame
