"""Verify local detector asset and record first/warm calls without downloads."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from perception.recorded_detector import RecordedDetector, verify_local_asset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "configs/models.yaml").read_text())
    fixture = manifest["fixtures"][0]
    fake = RecordedDetector(root / fixture["path"], fixture["sha256"])
    fake.close()
    model = manifest["models"][0]
    asset = verify_local_asset(root / model["path"], model["sha256"])
    if asset.stat().st_size != model["size_bytes"]:
        raise ValueError("Unexpected model size")
    # Dependency imported only after local assets are verified.
    import cv2

    from perception.object_detection import ObjectDetector
    from perception.types import PerceptionConfig

    image = cv2.imread(str(root / "src/perception/street.jpg"))
    if image is None:
        raise RuntimeError("Missing sample image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    report = dict(
        status="INCOMPLETE",
        asset=model,
        versions={k: importlib.metadata.version(k) for k in ("ultralytics", "torch", "numpy")},
        timing_scope="backend construction separately from first inference and three subsequent inferences; wall time, sample only, not accuracy",
    )
    start = time.perf_counter()
    detector = ObjectDetector(PerceptionConfig(model_name=str(asset)))
    report["construction_wall_s"] = time.perf_counter() - start
    try:
        durations = []
        counts = []
        for _ in range(4):
            start = time.perf_counter()
            result = detector.detect(image)
            durations.append(time.perf_counter() - start)
            counts.append(len(result))
        report.update(
            first_call_wall_s=durations[0],
            warm_calls_wall_s=durations[1:],
            detection_counts=counts,
            status="PASS",
        )
    finally:
        detector.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
