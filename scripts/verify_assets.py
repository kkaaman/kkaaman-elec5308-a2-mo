"""Check the model files in this clone, and optionally back them up offline.

    python scripts/verify_assets.py             # size + SHA256 of every model file
    python scripts/verify_assets.py --load      # also load the segmentation model on a sample
    python scripts/verify_assets.py --export D  # copy verified files + SHA256SUMS into folder D

Exit status is non-zero if any file is missing, still a Git LFS pointer, or
does not match its recorded checksum. Expected values are recorded in
docs/assets.md; BL-C-03 moves them into configs/models.yaml.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

from autonomy.paths import REPO_ROOT, SRC_DIR, model_path

#: name -> (file, size in bytes, SHA256). Matches the Git LFS pointers.
EXPECTED = {
    "segmentation": ("segmentation_model.pth", 9_505_431,
                     "a631440d3fe2ccd2554631bb49b1352c0efbbaffc2316caa79c288757e8dde2a"),
    "lane": ("lane_unet_best.pth", 57_385_466,
             "10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff"),
    "yolo": ("yolo26n.pt", 5_544_453,
             "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
}

SAMPLE_IMAGE = SRC_DIR / "perception" / "street.jpg"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str) -> tuple[bool, str]:
    """Return (ok, message) for one expected asset."""
    file_name, size, digest = EXPECTED[name]
    try:
        path = model_path(name)
    except FileNotFoundError as exc:
        return False, str(exc)
    actual_size = path.stat().st_size
    if actual_size != size:
        return False, f"{file_name}: size {actual_size} != expected {size}"
    actual = sha256(path)
    if actual != digest:
        return False, f"{file_name}: SHA256 {actual[:12]}… != expected {digest[:12]}…"
    return True, f"{file_name}: OK ({size:,} bytes, sha256 {digest[:12]}…)"


def load_segmentation() -> str:
    """Load the segmentation model and run it on the bundled sample image."""
    import cv2
    import numpy as np

    from segmentation import SemanticSegmenter

    image = cv2.imread(str(SAMPLE_IMAGE))
    if image is None:
        raise FileNotFoundError(f"sample image missing: {SAMPLE_IMAGE}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = SemanticSegmenter(str(model_path("segmentation")), device="cpu").predict(rgb)
    classes = np.unique(mask)
    road_share = float((mask == 0).mean())
    return (f"segmentation load: mask {mask.shape}, {len(classes)} classes present, "
            f"road (class 0) {road_share:.0%} of pixels")


def export(folder: Path) -> None:
    """Copy every verified file plus a SHA256SUMS file into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = []
    for name, (file_name, _, digest) in EXPECTED.items():
        shutil.copy2(model_path(name), folder / file_name)
        lines.append(f"{digest}  {file_name}")
    (folder / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--load", action="store_true", help="load the segmentation model on a sample image")
    parser.add_argument("--export", type=Path, metavar="FOLDER", help="copy verified files and SHA256SUMS here")
    args = parser.parse_args(argv)

    print(f"repository: {REPO_ROOT}")
    results = [check(name) for name in EXPECTED]
    for ok, message in results:
        print(("  ok   " if ok else "  FAIL ") + message)
    if not all(ok for ok, _ in results):
        return 1
    if args.load:
        print("  " + load_segmentation())
    if args.export:
        export(args.export)
        print(f"  exported {len(EXPECTED)} files and SHA256SUMS to {args.export}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
