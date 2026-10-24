"""Record the sensor frames a mapless drive saw, so the driver can be replayed offline.

Three live turn runs on Town10HD (24 Oct) placed the exit lane 1.1 m off, from a kerb
read at 27 m; the console and the drive log showed the symptom, not the cause. A replay
(``scripts/replay_mapless_frames.py``) runs the same driver code over the recorded frames
on the development machine, with the terrain grid, exit-line stations and truth to hand.

What is written under ``<run>/frames``:
- ``meta.json``: episode and calibration ids, the LiDAR's name and extrinsics, the suite.
- ``frames.jsonl``: one row per tick: frame id, times, speed, GNSS fix, IMU sample and the
  per-sensor status list (cameras and radars are not stored; the G1 driver uses neither).
- ``t<tick>.npz``: the LiDAR sweep as int16 centimetres (x, y, z; intensity is range-only
  in the simulator and unused), ~90 KB a tick against 250 KB as float32.

The replay is open loop: the car follows the logged motion whatever the replayed driver
commands, so it answers questions about perception and estimation, not control.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mapless.frame import GnssFix, ImuSample, LidarSweep, SensorFrame, SensorState, SensorStatus


class FrameRecorder:
    def __init__(self, directory, suite, calibration_id, episode_id, extrinsics):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lidar = suite.lidars[0].name
        (self.directory / "meta.json").write_text(
            json.dumps(
                dict(
                    schema="group7-mapless-frames-1",
                    episode_id=episode_id,
                    calibration_id=calibration_id,
                    lidar=self.lidar,
                    lidar_to_ego=np.asarray(extrinsics[self.lidar]).tolist(),
                    suite=suite.name,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        self._rows = open(self.directory / "frames.jsonl", "w", encoding="utf-8")
        self.ticks = 0

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
        )
        self._rows.write(json.dumps(row) + "\n")
        for sweep in frame.lidars:
            if sweep.name == self.lidar:
                np.savez_compressed(self.directory / f"t{tick:05d}.npz", xyz_cm=np.round(sweep.points[:, :3] * 100.0).astype(np.int16))
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
    with open(directory / "frames.jsonl", encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            lidars = ()
            if row["lidar"]:
                xyz = np.load(directory / f"t{row['tick']:05d}.npz")["xyz_cm"].astype(np.float32) / 100.0
                lidars = (LidarSweep(meta["lidar"], np.column_stack((xyz, np.zeros(len(xyz), dtype=np.float32))), extrinsics),)
            # Cameras and radars are not stored, so their statuses are dropped (a frame may
            # carry OK status only for data it holds); the G1 driver reads neither.
            status = tuple(
                SensorStatus(s["name"], s["kind"], SensorState(s["state"]), s["capture_time_s"], s.get("detail", ""))
                for s in row["status"]
                if s["kind"] not in ("camera", "radar")
            )
            frame = SensorFrame(
                episode_id=meta["episode_id"],
                frame_id=int(row["frame_id"]),
                sim_time_s=float(row["sim_time_s"]),
                received_monotonic_s=float(row["received_monotonic_s"]),
                cameras=(),
                lidars=lidars,
                radars=(),
                gnss=None if row["gnss"] is None else GnssFix(*row["gnss"]),
                imu=None if row["imu"] is None else ImuSample(row["imu"]["accel_mps2"], row["imu"]["gyro_rps"], row["imu"]["compass_rad"]),
                speed_mps=float(row["speed_mps"]),
                status=status,
                calibration_id=meta["calibration_id"],
            )
            yield int(row["tick"]), frame
