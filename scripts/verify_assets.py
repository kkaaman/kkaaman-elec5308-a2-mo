"""Verify Git LFS identity and offline model inference (RV-01/RV-02).

Run from any directory with the project installed. No weights are downloaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

FILES = {"lane": "lane_unet_best.pth", "segmentation": "segmentation_model.pth"}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pointer(text: str) -> tuple[str, int]:
    lines = text.splitlines()
    if not lines or lines[0] != "version https://git-lfs.github.com/spec/v1":
        raise ValueError("Committed file is not a Git LFS pointer")
    fields = dict(line.split(" ", 1) for line in lines[1:] if " " in line)
    oid = fields.get("oid", "")
    digest = oid.removeprefix("sha256:")
    if not oid.startswith("sha256:") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid SHA256 LFS object ID")
    size = int(fields.get("size", "0"))
    if size <= 0:
        raise ValueError("Invalid LFS object size")
    return digest, size


def verify_file(path: Path, pointer: str) -> dict:
    expected_hash, expected_size = parse_pointer(pointer)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path.name}; recover through authorized git lfs pull")
    with path.open("rb") as handle:
        if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError(f"{path.name} is still a pointer; run git lfs pull")
    size = path.stat().st_size
    actual_hash = sha256(path)
    if size != expected_size or actual_hash != expected_hash:
        raise ValueError(f"{path.name}: LFS size/checksum mismatch")
    return {"file": path.name, "size_bytes": size, "sha256": actual_hash, "identity": "PASS"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=FILES, default=list(FILES))
    parser.add_argument("--sample", type=Path, help="RGB source image loaded through OpenCV")
    parser.add_argument("--backup-dir", type=Path, required=True, help="Local offline copy destination")
    parser.add_argument("--output", type=Path, required=True, help="New evidence JSON; will not overwrite")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new evidence filename")
    report = {"utc": datetime.now(timezone.utc).isoformat(), "models": {}, "status": "FAIL"}
    try:
        report["source_commit"] = git("rev-parse", "HEAD")
        report["origin"] = git("remote", "get-url", "origin")
        report["tracked_changes"] = git("diff", "--name-only", "HEAD") .splitlines()
        from unittest.mock import patch

        import cv2
        import numpy as np
        import torch
        report["python"] = sys.version
        report["torch"] = torch.__version__
        torch.set_num_threads(2)
        sample_path = args.sample or ROOT / "src/perception/street.jpg"
        bgr = cv2.imread(str(sample_path))
        if bgr is None:
            raise ValueError(f"Could not read sample: {sample_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        report["sample"] = {"file": sample_path.name, "sha256": sha256(sample_path), "shape": list(rgb.shape)}
        for name in args.models:
            path = ROOT / FILES[name]
            record = verify_file(path, git("show", f"HEAD:{FILES[name]}"))
            report["models"][name] = record
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            backup = args.backup_dir / path.name
            if backup.resolve() == path.resolve():
                raise ValueError("Backup must be a separate copy")
            if backup.exists():
                verify_file(backup, git("show", f"HEAD:{FILES[name]}"))
            else:
                shutil.copy2(path, backup)
                verify_file(backup, git("show", f"HEAD:{FILES[name]}"))
            record["offline_copy"] = "PASS"
            record["backup_location"] = str(args.backup_dir)
            with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Unexpected weight download")):
                if name == "lane":
                    from perception.lane_inference import LaneSegmenter
                    model = LaneSegmenter(str(backup), device="cpu")
                    classes = 2
                else:
                    from segmentation.segmentation import SemanticSegmenter
                    model = SemanticSegmenter(str(backup), device="cpu")
                    classes = 19
                mask = model.predict(rgb)
            if mask.shape != rgb.shape[:2] or mask.dtype != np.uint8:
                raise AssertionError("Prediction shape/dtype contract failed")
            if np.any(mask >= classes):
                raise AssertionError("Prediction contains invalid labels")
            record["model_load_and_sample_inference"] = "PASS"
            record["labels_present"] = np.unique(mask).tolist()
            del model
        report["status"] = "PASS"
        report["scope"] = "Asset identity, offline copy and inference contract; not accuracy or driving acceptance."
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
