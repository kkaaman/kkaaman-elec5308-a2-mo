"""Real-model protocol smoke; black images establish service behavior, not accuracy."""

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle
from navigation_control.types import VehicleState
from perception.remote_detector import JpegHttpBackend


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/detect")
    parser.add_argument("--model-version", default=os.environ.get("DETECTOR_MODEL_SHA256"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    backends = [JpegHttpBackend(args.endpoint,source_id="remote",model_version=args.model_version,
                               timeout_s=60,run_id=f"smoke-client-{i}") for i in range(2)]
    now = time.monotonic()
    frame = FrameBundle("repeated-episode",0,0.,now,EgoPose(VehicleState(0,0,0,0)),
                        np.zeros((64,64,3),dtype=np.uint8),
                        CameraCalibration(64,64,32,32,((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))),
                        capture_started_monotonic_s=now)
    result = dict(origin="black-image-real-model-protocol-smoke",accuracy="NOT_MEASURED",
                  model_sha256=args.model_version,clients=[],duplicate_status=None,
                  repeated_episode_valid=False)
    try:
        for backend in backends:
            backend.reset(episode_id=frame.episode_id)
            batch = backend.detect(frame)
            result["clients"].append(dict(valid=batch.valid,source_id=batch.metadata.source_id,
                    run_id=backend.run_id,session_id=backend.session_id,
                    detections=len(batch.detections),timing=backend.last_timing))
        try:
            backends[0].detect(frame)
        except urllib.error.HTTPError as error:
            result["duplicate_status"] = error.code
        old = backends[0].session_id
        backends[0].reset(episode_id=frame.episode_id)
        batch = backends[0].detect(frame)
        result["repeated_episode_valid"] = batch.valid and backends[0].session_id != old
        result["passed"] = (all(c["valid"] for c in result["clients"])
                            and result["clients"][0]["session_id"] != result["clients"][1]["session_id"]
                            and result["duplicate_status"] == 409 and result["repeated_episode_valid"])
    finally:
        for backend in backends:
            backend.close()
    text = json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(text+"\n")
    print(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
