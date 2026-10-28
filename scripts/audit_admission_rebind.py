"""Read-only admission audit; negative probes mutate isolated registered copies only."""

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from autonomy import admission
from autonomy.baseline_operation import frozen_governor_config
from perception.recorded_detector import verify_local_asset


def audit(root):
    root = Path(root).resolve()
    mission = root / "configs/mission_baseline.json"
    record = admission.read_json(root / admission.RECORD_PATH)
    original_root = admission.ADMISSION_ROOT
    probes = {}
    try:
        admission.ADMISSION_ROOT = root
        baseline = admission.load_admitted_baseline(mission)
        config = frozen_governor_config(mission)
        assert (config.cruise_mps,config.deceleration_mps2,config.response_budget_s,config.stop_margin_m) == (
            4.,1.497566177392398,.9500000052154064,6.)
        mutations = {
            "candidate": lambda d:d["baseline_threshold_freeze"].update(status="CANDIDATE"),
            "cruise": lambda d:d["baseline_threshold_freeze"]["frozen_values"].update(cruise_speed_mps=4.5),
            "deceleration": lambda d:d["baseline_threshold_freeze"]["frozen_values"].update(deceleration_mps2=1.6),
            "response": lambda d:d["baseline_threshold_freeze"]["frozen_values"].update(response_budget_s=1.),
            "margin": lambda d:d["baseline_threshold_freeze"]["frozen_values"].update(stop_margin_m=5.9),
            "scope": lambda d:d["baseline_threshold_freeze"].update(operational_scope="General/pedestrian"),
            "profile": lambda d:d["baseline_threshold_freeze"].update(admitted_profile="general"),
            "model_identity": lambda d:d["baseline_threshold_freeze"]["detector"].update(model_sha256="0"*64),
            "sensor_rig": lambda d:d["baseline_threshold_freeze"]["camera_rig"].update(width=1),
            "confirmation": lambda d:d["baseline_threshold_freeze"].update(track_policy={}),
            "depth_policy": lambda d:d["baseline_threshold_freeze"].update(depth_policy={}),
            "runner_hash": lambda d:d["baseline_threshold_freeze"].update(issue41_runner_scope_sha256="0"*64),
            "runner_revision": lambda d:d["baseline_threshold_freeze"].update(issue41_runner_source_revision="0"*40),
        }
        names = list(mutations)+["changed_source","missing_source","added_source","aux_config","record_checksum"]
        with tempfile.TemporaryDirectory(prefix="g1-negative-") as directory:
            base = Path(directory)/"base"
            for name in record["source_variants"]["canonical_loader_only"]["source_hashes"]:
                target=base/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/name,target)
            for name in (admission.RECORD_PATH,"configs/mission_baseline.json"):
                target=base/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/name,target)
            for name in names:
                tree=Path(directory)/name
                shutil.copytree(base,tree)
                path=tree/"configs/mission_baseline.json"
                if name in mutations:
                    doc=admission.read_json(path)
                    mutations[name](doc)
                    path.write_text(json.dumps(doc))
                elif name=="changed_source":
                    target=tree/"src/autonomy/stack.py"
                    target.write_bytes(target.read_bytes()+b"\n# drift\n")
                elif name=="missing_source":
                    (tree/"src/autonomy/stack.py").unlink()
                elif name=="added_source":
                    (tree/"src/autonomy/unreviewed_probe.py").write_text("# unreviewed\n")
                elif name=="aux_config":
                    target=tree/"configs/acceptance.json"
                    target.write_bytes(target.read_bytes()+b"\n")
                else:
                    target=tree/admission.RECORD_PATH
                    doc=admission.read_json(target)
                    doc["config_identity_sha256"]="0"*64
                    target.write_text(json.dumps(doc))
                admission.ADMISSION_ROOT=tree
                try:
                    frozen_governor_config(path)
                except (ValueError,FileNotFoundError) as error:
                    probes[name]=dict(status="REJECTED",reason=str(error))
                else:
                    raise AssertionError(f"Negative probe accepted: {name}")
            asset=Path(directory)/"model.bin"
            asset.write_bytes(b"pinned local model")
            expected=hashlib.sha256(asset.read_bytes()).hexdigest()
            verify_local_asset(asset,expected)
            asset.write_bytes(b"altered local model")
            try:
                verify_local_asset(asset,expected)
            except ValueError as error:
                probes["actual_model_bytes"]=dict(status="REJECTED",reason=str(error))
            else:
                raise AssertionError("Altered model bytes accepted")
    finally:
        admission.ADMISSION_ROOT=original_root
    return dict(status="PASS",configuration_semantic_sha256=admission.config_identity(baseline),
                configuration_file_sha256=hashlib.sha256(mission.read_bytes()).hexdigest(),
                source_fingerprint=record["source_variants"]["canonical_loader_only"]["scope_sha256"],
                record_sha256=hashlib.sha256((root/admission.RECORD_PATH).read_bytes()).hexdigest(),
                negative_probes=probes,production_tree_mutated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report,stream,indent=2)
        stream.write("\n")
    print(f"PASS: production admission and {len(report['negative_probes'])} negative probes")


if __name__=="__main__":
    main()
