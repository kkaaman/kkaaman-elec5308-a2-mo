"""One reproducible scenario; CARLA and harness use the same production policy."""

import argparse
import json
from pathlib import Path

from autonomy.configuration import content_hash, read_config
from evaluation.run_logger import file_sha256, source_identity
from evaluation.runner import run_episode

ROOT = Path(__file__).resolve().parents[1]


def arguments(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("harness", "carla"), default="harness")
    parser.add_argument("--scenario", choices=tuple(f"S{x}" for x in range(6)), default="S0")
    parser.add_argument("--seed", type=int, default=5308)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baseline_stack.json")
    parser.add_argument("--mission", type=Path)
    parser.add_argument("--depth-calibration", type=Path)
    parser.add_argument(
        "--event-verification",
        type=Path,
        help="BL-B-07 live report; without it collision/lane metrics stay unscored",
    )
    parser.add_argument(
        "--declared-crossings", type=Path, default=ROOT / "configs/declared_crossings.json"
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo26n.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--navigation-only", action="store_true")
    parser.add_argument("--max-sim-s", type=float, default=180)
    parser.add_argument(
        "--harness-route",
        choices=("mission", "fixture"),
        default="mission",
        help="Harness only: real Town10HD mission with flat-ground depth, or the synthetic fixture",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def execute(args):
    if args.seed < 0 or not 1 <= args.max_sim_s <= 600:
        raise ValueError("Invalid seed/time budget")
    if args.navigation_only and args.scenario != "S0":
        raise ValueError("Navigation-only runs must use clear S0")
    if args.output.exists():
        raise FileExistsError(f"Choose a new run folder: {args.output}")
    config, document, digest = read_config(args.config)
    manifest = dict(
        source_identity(ROOT),
        engine=args.engine,
        scenario=args.scenario,
        seed=args.seed,
        configuration_status=document["status"],
        config_sha256=digest,
        mode="navigation" if args.navigation_only else "full_stack",
    )
    if args.engine == "harness":
        from sim_harness.baseline import BaselineWorld, RasterDetector

        world = BaselineWorld(
            args.scenario, curve=True, seed=args.seed, route=args.harness_route
        )
        detector = RasterDetector()
        points = world.points
        manifest.update(
            harness_route=args.harness_route,
            step_wall_budget_s=10.0,
            event_sensors_verified=True,
            truth_channel_complete=True,
            model_sha256="synthetic-colour-v1",
            mission_sha256=content_hash(points),
        )
    else:
        import carla

        from carla_adapter.baseline_world import CarlaWorld
        from evaluation.event_checks import load_declarations
        from perception.frame_detector import EmptyDetector, FrameDetector
        from planning.mission import Mission

        if args.mission is None or args.depth_calibration is None:
            raise ValueError("CARLA requires saved --mission and --depth-calibration")
        mission = Mission.load(args.mission)
        depth = json.loads(args.depth_calibration.read_text(encoding="utf-8-sig"))
        if (
            depth.get("status") != "MEASURED_PASS"
            or depth.get("opendrive_sha256") != mission.opendrive_sha256
        ):
            raise ValueError("Depth convention must be measured for the same map build")
        if document["status"] == "FROZEN_FOR_VALIDATION":
            proof = document["evidence"]
            if (
                proof.get("source_sha256") != manifest["source_sha256"]
                or proof.get("mission_sha256") != mission.sha256
            ):
                raise ValueError(
                    "Source or mission changed after configuration freeze; revalidate the candidate"
                )
            if proof.get("depth", {}).get("sha256") != file_sha256(args.depth_calibration):
                raise ValueError("Depth calibration changed after configuration freeze")
        scenario_path = ROOT / f"configs/scenarios/{args.scenario}.yaml"
        scenario = json.loads(scenario_path.read_text())
        verification = None
        if args.event_verification is not None:
            verification = json.loads(args.event_verification.read_text(encoding="utf-8-sig"))
            manifest["event_verification_sha256"] = file_sha256(args.event_verification)
        declarations = load_declarations(
            args.declared_crossings if args.declared_crossings.is_file() else None
        )
        if args.declared_crossings.is_file():
            manifest["declared_crossings_sha256"] = file_sha256(args.declared_crossings)
        client = carla.Client(args.host, args.port)
        client.set_timeout(60)
        world = CarlaWorld(
            client,
            mission,
            scenario,
            config,
            depth_convention=depth["convention"],
            output=args.output,
            follow=args.follow,
            event_verification=verification,
            declared_crossings=declarations,
        )
        points = mission.points_xy_m
        manifest.update(
            mission_sha256=mission.sha256,
            depth_calibration_sha256=file_sha256(args.depth_calibration),
            scenario_sha256=file_sha256(scenario_path),
            step_wall_budget_s=0.1,
        )
        if args.navigation_only:
            detector = EmptyDetector()
            manifest["model_sha256"] = "navigation-only-no-model"
        else:
            import torch
            import ultralytics

            from perception import ObjectDetector, PerceptionConfig
            from perception.recorded_detector import verify_local_asset

            asset = json.loads((ROOT / "configs/models.yaml").read_text())["models"][0]
            verify_local_asset(args.weights, asset["sha256"])
            detector = FrameDetector(
                ObjectDetector(PerceptionConfig(model_name=str(args.weights.resolve()))),
                model_version=asset["sha256"],
            )
            manifest.update(
                model_sha256=asset["sha256"],
                torch=torch.__version__,
                ultralytics=ultralytics.__version__,
                cuda_available=torch.cuda.is_available(),
            )
    result = run_episode(
        world,
        detector,
        points,
        config,
        args.output,
        manifest,
        max_sim_s=args.max_sim_s,
        navigation_only=args.navigation_only,
    )
    print(f"{args.scenario}: {result['collection_status']} | {args.output}", flush=True)
    if result["error"]:
        print(result["error"], flush=True)
    return result


def main():
    result = execute(arguments().parse_args())
    return 0 if result["collection_status"] == "COLLECTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
