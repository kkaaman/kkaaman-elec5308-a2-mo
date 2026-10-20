"""Recompute S0 acceptance solely from persisted run files."""
import argparse
import json
from pathlib import Path

from evaluation.metrics import aggregate
from evaluation.run_log import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(args.runs)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
