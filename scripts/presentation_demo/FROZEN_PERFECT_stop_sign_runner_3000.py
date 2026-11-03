import copy
import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from navigation_control import Waypoint
from navigation_control.types import ControlCommand
from autonomy.mission import (
    MissionEpisode as BaseMissionEpisode,
)
from planning.baseline_mission import (
    load_frozen_mission,
)
from carla_adapter.s0_runner import (
    run_s0_episode,
)
from carla_adapter.world import (
    SynchronousWorldSession,
)
import carla_adapter.s0_runner as s0_runner


ROOT = Path(
    "/home/alan/release-candidate-20261031"
)

ROUTE_PATH = Path(
    "/home/alan/elec5308-demo-frozen/"
    "town10hd_demo_long.json"
)

STOP_TARGET_FILE = Path(
    "/home/alan/elec5308-demo-frozen/"
    "stop_target.txt"
)

STOP_DONE_FILE = Path(
    "/home/alan/elec5308-demo-frozen/"
    "stop_done.flag"
)

OUTPUT = (
    ROOT
    / "runs/release/final-demo-persistent-01"
)

STOP_TARGET_FILE.unlink(
    missing_ok=True
)

STOP_DONE_FILE.unlink(
    missing_ok=True
)

if OUTPUT.exists():
    raise SystemExit(
        f"Output already exists: {OUTPUT}"
    )


baseline = load_frozen_mission(
    ROOT / "configs/mission_baseline.json"
)

raw = json.loads(
    ROUTE_PATH.read_text()
)

points = [
    (
        float(x),
        -float(y),
        float(z),
    )
    for x, y, z
    in raw["points_xyz_m"]
]

if len(points) < 3:
    raise RuntimeError(
        "Demo route contains too few points"
    )


x0, y0, _ = points[0]
x1, y1, _ = points[1]

heading = math.atan2(
    y1 - y0,
    x1 - x0,
)

REAR_AXLE_OFFSET_M = 1.38641

spawn_pose = {
    "x_m": (
        x0
        + REAR_AXLE_OFFSET_M
        * math.cos(heading)
    ),
    "y_m": (
        y0
        + REAR_AXLE_OFFSET_M
        * math.sin(heading)
    ),
    "z_m": 0.6,
    "yaw_rad": heading,
}

route_identity = {
    "route_kind":
        "DEMO_ONLY_NOT_ACCEPTANCE",
    "map_name": baseline.map_name,
    "points_xyz_m": points,
    "spawn_pose": spawn_pose,
}

route_sha256 = hashlib.sha256(
    json.dumps(
        route_identity,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()

config = copy.deepcopy(
    baseline.config
)

config["runtime"][
    "max_simulation_seconds"
] = 400.0

config["demo_only"] = True

config["demo_description"] = (
    "Persistent Town10HD live demonstration"
)

config_sha256 = hashlib.sha256(
    json.dumps(
        {
            "config": config,
            "route_sha256": route_sha256,
            "demo_only": True,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


class DemoMission:
    def __init__(self):
        self.config = config
        self.config_sha256 = (
            config_sha256
        )
        self.route_sha256 = (
            route_sha256
        )

        self.controller_config = (
            baseline.controller_config
        )

        self.waypoints = tuple(
            Waypoint(x, y)
            for x, y, _ in points
        )

        self.spawn_pose = spawn_pose
        self.map_name = baseline.map_name
        self.opendrive_sha256 = (
            baseline.opendrive_sha256
        )


class StopLineMissionEpisode(
    BaseMissionEpisode
):
    """Perception-armed stop-line profile."""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self._stop_active = False
        self._stopped_since = None
        self._waiting_for_clear = False
        self._last_report_bucket = None

    def step(
        self,
        state,
        *args,
        **kwargs,
    ):
        # Always calculate the normal navigation
        # command first.
        normal_result = super().step(
            state,
            *args,
            **kwargs,
        )

        now = time.monotonic()

        # After a completed stop, immediately allow
        # normal navigation while waiting for the
        # viewer to clear/rearm the old sign.
        if self._waiting_for_clear:
            if (
                not STOP_TARGET_FILE.exists()
                and not STOP_DONE_FILE.exists()
            ):
                self._waiting_for_clear = False

                print(
                    "STOP CONTROLLER REARMED",
                    flush=True,
                )

            return normal_result

        try:
            line_distance = float(
                STOP_TARGET_FILE
                .read_text()
                .strip()
            )

        except Exception:
            return normal_result

        speed_mps = float(
            getattr(
                state,
                "speed_mps",
                0.0,
            )
        )

        # This was the calibrated profile that
        # stopped at the correct junction line.
        STOP_MARGIN_M = 0.35
        PROFILE_DECEL_MPS2 = 0.85

        remaining = max(
            0.0,
            line_distance
            - STOP_MARGIN_M,
        )

        desired_speed = (
            2.0
            * PROFILE_DECEL_MPS2
            * remaining
        ) ** 0.5

        desired_speed = min(
            desired_speed,
            4.0,
        )

        speed_error = (
            speed_mps
            - desired_speed
        )

        bucket = int(
            max(
                0.0,
                line_distance,
            )
        )

        if (
            bucket
            != self._last_report_bucket
        ):
            print(
                "STOP PROFILE "
                f"line={line_distance:.2f}m "
                f"target="
                f"{desired_speed:.2f}m/s "
                f"actual="
                f"{speed_mps:.2f}m/s",
                flush=True,
            )

            self._last_report_bucket = (
                bucket
            )

        # Far enough away that normal cruise is
        # still compatible with the stop profile.
        if (
            desired_speed >= 3.9
            and not self._stop_active
        ):
            return normal_result

        self._stop_active = True

        command = normal_result.command

        if speed_error > 0.80:
            brake = 0.45

        elif speed_error > 0.40:
            brake = 0.28

        elif speed_error > 0.15:
            brake = 0.14

        elif speed_error > -0.10:
            brake = 0.06

        else:
            brake = 0.0

        result = replace(
            normal_result,
            command=ControlCommand(
                steering_angle_rad=(
                    command.steering_angle_rad
                ),
                throttle=0.0,
                brake=brake,
            ),
        )

        if remaining <= 0.20:
            result = replace(
                result,
                command=ControlCommand(
                    steering_angle_rad=(
                        command
                        .steering_angle_rad
                    ),
                    throttle=0.0,
                    brake=max(
                        0.55,
                        result.command.brake,
                    ),
                ),
            )

        # Vehicle has actually come to rest.
        # Do NOT require an exact tiny distance
        # window here.
        if (
            self._stop_active
            and speed_mps <= 0.10
        ):
            if (
                self._stopped_since
                is None
            ):
                self._stopped_since = now

                print(
                    "STOP LINE: FULL STOP "
                    f"line="
                    f"{line_distance:.2f}m "
                    "- HOLDING 2.0s",
                    flush=True,
                )

            # Explicit stationary hold.
            result = replace(
                result,
                command=ControlCommand(
                    steering_angle_rad=(
                        command
                        .steering_angle_rad
                    ),
                    throttle=0.0,
                    brake=max(
                        0.55,
                        result.command.brake,
                    ),
                ),
            )

            if (
                now
                - self._stopped_since
                >= 2.0
            ):
                self._stop_active = False
                self._stopped_since = None
                self._waiting_for_clear = True

                STOP_DONE_FILE.write_text(
                    "DONE\n"
                )

                print(
                    "STOP LINE: "
                    "2.0s HOLD COMPLETE "
                    "- RESUMING",
                    flush=True,
                )

                # Release the stop override now,
                # not one control cycle later.
                return normal_result

        else:
            self._stopped_since = None

        return result


s0_runner.MissionEpisode = (
    StopLineMissionEpisode
)

mission = DemoMission()

print("FINAL PERSISTENT DEMO")
print("=====================")
print("map:", mission.map_name)
print("points:", len(points))

length = sum(
    math.hypot(
        b[0] - a[0],
        b[1] - a[1],
    )
    for a, b
    in zip(
        points,
        points[1:],
    )
)

print(f"length: {length:.1f} m")
print(
    f"heading: "
    f"{math.degrees(heading):.1f} deg"
)
print("output:", OUTPUT)
print()
print(
    "DEMO ONLY - "
    "NOT ACCEPTANCE EVIDENCE"
)
print()


_original_tick = (
    SynchronousWorldSession.tick
)


def realtime_tick(
    self,
    *args,
    **kwargs,
):
    result = _original_tick(
        self,
        *args,
        **kwargs,
    )

    time.sleep(0.015)

    return result


SynchronousWorldSession.tick = (
    realtime_tick
)

summary = run_s0_episode(
    mission,
    OUTPUT,
    host="172.28.176.1",
    port=3000,
    navigation_only=True,
    mission_path=None,
)

print()
print("DEMO FINISHED")
print(
    json.dumps(
        summary,
        indent=2,
    )
)
