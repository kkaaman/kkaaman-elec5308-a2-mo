#!/usr/bin/env python3
"""Verify model weight files against ``assets/model_manifest.json``.

For each selected asset this checks that:

1. the file exists and is real weights, not a Git LFS pointer;
2. its size and SHA-256 match the manifest (copied from the LFS pointer);
3. optionally (``--load``), it loads into its model and runs one inference.

The exit code is 0 only when every requested check passes.

Examples::

    # Hash-check every asset in the manifest
    python scripts/verify_model_asset.py

    # RV-01: hash, load and run the segmentation model; save a provenance record
    python scripts/verify_model_asset.py segmentation_bisenetv2 --load \
        --sample src/perception/street.jpg --record runs/assets/rv01.json

    # Offline backup: copy every verified file plus a SHA256SUMS file
    python scripts/verify_model_asset.py --export D:/backup/elec5308-models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "assets" / "model_manifest.json"

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
LFS_POINTER_MAX_BYTES = 1024

CITYSCAPES_CLASSES = (
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
)


@dataclass
class AssetResult:
    """Outcome of checking one manifest entry."""

    name: str
    path: str
    status: str  # ok | missing | lfs_pointer | size_mismatch | hash_mismatch
    message: str
    expected_size: int
    expected_sha256: str
    actual_size: int | None = None
    actual_sha256: str | None = None
    load: dict[str, Any] | None = field(default=None)

    @property
    def passed(self) -> bool:
        load_ok = self.load is None or self.load.get("status") in ("ok", "skipped")
        return self.status == "ok" and load_ok


def parse_lfs_pointer(path: Path) -> dict[str, Any] | None:
    """Return ``{"oid": ..., "size": ...}`` if ``path`` is a Git LFS pointer, else None."""
    if path.stat().st_size > LFS_POINTER_MAX_BYTES:
        return None
    data = path.read_bytes()
    if not data.startswith(LFS_POINTER_PREFIX):
        return None
    fields: dict[str, Any] = {}
    for line in data.decode("utf-8", errors="replace").splitlines():
        key, _, value = line.partition(" ")
        if key == "oid" and value.startswith("sha256:"):
            fields["oid"] = value.removeprefix("sha256:")
        elif key == "size" and value.isdigit():
            fields["size"] = int(value)
    return fields


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_asset(entry: dict[str, Any], repo_root: Path) -> AssetResult:
    """Check existence, LFS-pointer status, size and hash for one manifest entry."""
    path = repo_root / entry["path"]
    result = AssetResult(
        name=entry["name"],
        path=entry["path"],
        status="ok",
        message="",
        expected_size=int(entry["size"]),
        expected_sha256=entry["sha256"].lower(),
    )

    if not path.is_file():
        result.status = "missing"
        result.message = f"{entry['path']} not found under {repo_root}"
        return result

    result.actual_size = path.stat().st_size
    pointer = parse_lfs_pointer(path)
    if pointer is not None:
        result.status = "lfs_pointer"
        result.message = (
            f"{entry['path']} is a {result.actual_size}-byte Git LFS pointer, not the weights. "
            f'Run: git lfs install && git lfs pull --include="{entry["path"]}"'
        )
        if pointer.get("oid") and pointer["oid"] != result.expected_sha256:
            result.message += " (pointer oid also differs from the manifest)"
        return result

    if result.actual_size != result.expected_size:
        result.status = "size_mismatch"
        result.message = (
            f"size {result.actual_size} bytes, expected {result.expected_size}; "
            "wrong file or incomplete download"
        )
        return result

    result.actual_sha256 = sha256_file(path)
    if result.actual_sha256 != result.expected_sha256:
        result.status = "hash_mismatch"
        result.message = "size matches but SHA-256 differs; this is not the committed checkpoint"
        return result

    result.message = "size and SHA-256 match the manifest"
    return result


def _load_rgb(sample: Path | None) -> tuple[Any, str]:
    """Return an HxWx3 uint8 RGB image and a description of where it came from."""
    import numpy as np

    if sample is not None:
        import cv2

        bgr = cv2.imread(str(sample), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not read sample image {sample}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), str(sample)

    # Synthetic fallback: proves the forward pass runs, says nothing about accuracy.
    height, width = 360, 640
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[: height // 2] = (135, 180, 230)  # sky-ish
    image[height // 2 :] = (90, 90, 90)  # road-ish
    return image, "synthetic 640x360 image (shape check only)"


def load_segmentation(weights: Path, sample: Path | None, repo_root: Path) -> dict[str, Any]:
    """Load the A1 BiSeNetV2 wrapper with ``weights`` and run one timed inference."""
    src = str(repo_root / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    import numpy as np
    import torch

    from segmentation import SemanticSegmenter

    segmenter = SemanticSegmenter(str(weights))  # strict state-dict load
    image, image_source = _load_rgb(sample)

    segmenter.predict(image)  # warm-up
    if segmenter.device == "cuda":
        torch.cuda.synchronize()
    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        mask = segmenter.predict(image)
    if segmenter.device == "cuda":
        torch.cuda.synchronize()
    mean_ms = (time.perf_counter() - start) * 1000.0 / runs

    if mask.shape != image.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")

    values, counts = np.unique(mask, return_counts=True)
    fractions = counts / counts.sum()
    order = np.argsort(fractions)[::-1][:6]
    top_classes = [
        {
            "class_id": int(values[i]),
            "class": CITYSCAPES_CLASSES[values[i]] if values[i] < len(CITYSCAPES_CLASSES) else "?",
            "fraction": round(float(fractions[i]), 4),
        }
        for i in order
    ]
    warnings = []
    if fractions.max() > 0.95:
        warnings.append("one class covers >95% of pixels; check the model on a real street frame")

    return {
        "status": "ok",
        "torch": torch.__version__,
        "device": segmenter.device,
        "input_size_wh": list(segmenter.input_size),
        "num_classes": segmenter.num_classes,
        "image_source": image_source,
        "mask_shape": list(mask.shape),
        "mean_inference_ms": round(mean_ms, 2),
        "timed_runs": runs,
        "top_classes": top_classes,
        "warnings": warnings,
    }


LOADERS: dict[str, Callable[[Path, Path | None, Path], dict[str, Any]]] = {
    "segmentation": load_segmentation,
}


def run_load(entry: dict[str, Any], sample: Path | None, repo_root: Path) -> dict[str, Any]:
    """Run the entry's loader, converting any failure into a reported result."""
    loader_name = entry.get("loader")
    if not loader_name:
        return {"status": "skipped", "reason": "no loader defined for this asset"}
    loader = LOADERS.get(loader_name)
    if loader is None:
        return {"status": "failed", "error": f"unknown loader '{loader_name}'"}
    try:
        return loader(repo_root / entry["path"], sample, repo_root)
    except Exception as exc:  # noqa: BLE001 - report every load failure
        error = f"{type(exc).__name__}: {exc}"
        hint = None
        if "weights_only" in str(exc):
            hint = (
                "Checkpoint holds non-tensor objects; re-save it as a plain state_dict "
                "rather than loading with weights_only=False."
            )
        elif isinstance(exc, ImportError):
            hint = "Install the project dependencies (pip install -e .) in this environment."
        return {"status": "failed", "error": error, "hint": hint}


def _git_state(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def build_record(results: list[AssetResult], manifest_path: Path, repo_root: Path) -> dict:
    """Provenance record for docs/assets.md and run evidence."""
    try:
        manifest_display = str(manifest_path.relative_to(repo_root))
    except ValueError:
        manifest_display = str(manifest_path)
    return {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": _git_state(repo_root),
        "manifest": manifest_display,
        "all_passed": all(r.passed for r in results),
        "assets": [asdict(r) for r in results],
    }


def export_assets(results: list[AssetResult], repo_root: Path, folder: Path) -> Path:
    """Copy verified files into ``folder`` with a ``SHA256SUMS`` file (offline copy)."""
    failed = [r.name for r in results if r.status != "ok"]
    if failed:
        raise ValueError(f"not exporting unverified assets: {', '.join(failed)}")
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for result in results:
        name = Path(result.path).name
        shutil.copy2(repo_root / result.path, folder / name)
        lines.append(f"{result.actual_sha256}  {name}")
    sums = folder / "SHA256SUMS"
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sums


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="asset names to check (default: all)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--load", action="store_true", help="also load and run each model")
    parser.add_argument("--sample", type=Path, help="RGB image for the --load inference")
    parser.add_argument("--record", type=Path, help="write a JSON provenance record here")
    parser.add_argument("--export", type=Path, metavar="FOLDER",
                        help="copy the verified files and a SHA256SUMS file here (offline backup)")
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["assets"]
    if args.names:
        known = {e["name"] for e in entries}
        unknown = sorted(set(args.names) - known)
        if unknown:
            parser.error(
                f"unknown asset(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}"
            )
        entries = [e for e in entries if e["name"] in args.names]

    results = []
    for entry in entries:
        result = check_asset(entry, args.repo_root)
        if args.load and result.status == "ok":
            result.load = run_load(entry, args.sample, args.repo_root)
        results.append(result)

        mark = "PASS" if result.passed else "FAIL"
        print(f"[{mark}] {result.name}: {result.message}")
        if result.load is not None:
            load = result.load
            if load["status"] == "ok":
                classes = ", ".join(
                    f"{c['class']} {c['fraction']:.0%}" for c in load["top_classes"]
                )
                print(
                    f"       load ok on {load['device']} (torch {load['torch']}), "
                    f"{load['mean_inference_ms']} ms/frame, mask {load['mask_shape']}"
                )
                print(f"       classes on {load['image_source']}: {classes}")
                for warning in load["warnings"]:
                    print(f"       warning: {warning}")
            else:
                print(f"       load {load['status']}: {load.get('error') or load.get('reason')}")
                if load.get("hint"):
                    print(f"       hint: {load['hint']}")

    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        record = build_record(results, args.manifest.resolve(), args.repo_root.resolve())
        args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"record written to {args.record}")

    if args.export:
        try:
            sums = export_assets(results, args.repo_root, args.export)
            print(f"exported {len(results)} file(s) and {sums.name} to {args.export}")
        except ValueError as exc:
            print(f"export skipped: {exc}")

    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
