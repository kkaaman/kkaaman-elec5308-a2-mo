"""Check hashes/alignment/split leakage and build a local RGB-label gallery."""

import argparse
import html
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from evaluation.dataset import local_file, read_dataset, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    root = args.dataset.resolve()
    manifest, rows = read_dataset(root)
    counts = Counter(row["split"] for row in rows)
    positive = Counter(row["split"] for row in rows if row["lane_pixels"] > 0)
    report = {
        "status": "STRUCTURE_CHECKED",
        "sample_count": len(rows),
        "splits": dict(counts),
        "frames_with_road_markings": dict(positive),
        "labels": "RoadLine paint; may include arrows, stop lines and crosswalk markings",
        "capture_scope": manifest["scope"],
        "needs_visual_review": True,
    }
    save_json(root / "dataset_review.json", report)
    previews = root / "previews"
    previews.mkdir(exist_ok=True)
    cards = []
    for split in ("train", "val", "test"):
        subset = [row for row in rows if row["split"] == split]
        stride = max(1, len(subset) // 12)
        for row in subset[::stride][:12]:
            with (
                Image.open(local_file(root, row["rgb"])) as image,
                Image.open(local_file(root, row["lane"])) as label,
            ):
                rgb = np.array(image)
                mask = np.asarray(label).astype(bool)
                overlay = rgb.copy()
                overlay[mask] = (0.35 * rgb[mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
                preview = previews / f"{row['sample_id']}.jpg"
                Image.fromarray(overlay).save(preview)
            caption = html.escape(
                f"{split} / {row['run_id']} / frame {row['frame_id']} / {row['lane_pixels']} labelled pixels"
            )
            cards.append(
                f'<figure><figcaption>{caption}</figcaption><img src="{html.escape(row["rgb"], quote=True)}"><img src="{html.escape(preview.relative_to(root).as_posix(), quote=True)}"></figure>'
            )
    title = "CARLA dataset review"
    document = (
        '<!doctype html><meta charset="utf-8"><title>'
        + title
        + "</title><style>body{font:16px system-ui;margin:24px;background:#eef2f4}figure{background:white;padding:12px;margin:16px 0}img{width:48%;margin:1%}figcaption{padding:8px}h1{font-size:26px}</style><h1>"
        + title
        + "</h1><p>Left: RGB. Right: same-frame RoadLine labels. Check alignment, incorrect tags, floating targets, dark/empty images and road-paint ambiguity before a long training run. These images do not establish driving performance.</p>"
        + "".join(cards)
    )
    (root / "review.html").write_text(document, encoding="utf-8")
    print(report)
    print("Open:", root / "review.html")


if __name__ == "__main__":
    main()
