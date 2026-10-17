"""Run every offline S0 replicate and retain all attempts, including failures."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation.metrics import aggregate
from evaluation.run_log import write_json
from sim_harness.s0_acceptance import run_s0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/S0"))
    parser.add_argument("--config", type=Path, default=Path("configs/acceptance.json"))
    parser.add_argument("--seed", type=int, default=5308)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directories = []
    for trial in range(config["trials"]):
        directory = args.output / f"{batch}-{trial}"
        run_s0(directory, config, args.seed + trial, root)
        directories.append(directory)
    result = aggregate(directories)
    write_json(args.output / f"{batch}-aggregate.json", result)
    print(json.dumps({"status": result["status"], "runs": [str(d) for d in directories]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
