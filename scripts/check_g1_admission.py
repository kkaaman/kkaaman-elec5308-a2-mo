"""Read-only G1 identity/loader check; never connects to CARLA or writes config."""

import argparse
import hashlib
import json
from pathlib import Path

from autonomy.admission import RECORD_PATH, config_identity, load_admitted_baseline, read_json
from autonomy.baseline_operation import frozen_governor_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/mission_baseline.json"
    document = load_admitted_baseline(path)
    record = read_json(root / RECORD_PATH)
    config = frozen_governor_config(path)
    report = dict(status="IDENTITY_AND_LOADER_PASS", config_identity_sha256=config_identity(document),
                  config_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  runner_scope_sha256=record["runner_scope_sha256"],
                  runner_base_revision=record["runner_base_revision"],
                  publication_state=record["publication_state"], cruise_mps=config.cruise_mps,
                  admitted_profile=record["admitted_profile"], live_acceptance=False)
    if args.output is not None:
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
