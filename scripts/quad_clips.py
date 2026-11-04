"""Four clips side by side in one 1920 x 1080 video (2 x 2), each labelled.

    python scripts/quad_clips.py A.mp4 B.mp4 C.mp4 D.mp4 --labels "Right turn" "Left turn" "Rain" "Parking" --output quad_clips.mp4

Top left, top right, bottom left, bottom right in that order. Each clip is scaled to
960 x 540 (letterboxed); a clip that ends first holds its last frame until the longest
ends. The output runs at --fps (default 20, the simulation's rate). Needs only OpenCV.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

TILE_W, TILE_H = 960, 540


def fit(frame):
    """The frame scaled into a 960 x 540 tile, letterboxed."""
    h, w = frame.shape[:2]
    scale = min(TILE_W / w, TILE_H / h)
    resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    y, x = (TILE_H - resized.shape[0]) // 2, (TILE_W - resized.shape[1]) // 2
    tile[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return tile


def label(tile, text):
    if text:
        cv2.rectangle(tile, (0, 0), (16 + 14 * len(text), 36), (0, 0, 0), -1)
        cv2.putText(tile, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs=4, help="four videos: top left, top right, bottom left, bottom right")
    parser.add_argument("--labels", nargs=4, default=None, help="a label for each clip (in the same order)")
    parser.add_argument("--output", default="quad_clips.mp4")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args(argv)
    captures = [cv2.VideoCapture(p) for p in args.clips]
    for path, cap in zip(args.clips, captures):
        if not cap.isOpened():
            parser.error(f"cannot open {path}")
    labels = args.labels or [""] * 4
    last = [np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8) for _ in captures]
    done = [False] * 4
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (2 * TILE_W, 2 * TILE_H))
    frames = 0
    while True:
        for k, cap in enumerate(captures):
            if not done[k]:
                ok, frame = cap.read()
                if ok:
                    last[k] = label(fit(frame), labels[k])
                else:
                    done[k] = True
        if all(done):
            break
        writer.write(np.vstack([np.hstack([last[0], last[1]]), np.hstack([last[2], last[3]])]))
        frames += 1
    writer.release()
    for cap in captures:
        cap.release()
    print(f"wrote {args.output}: {frames} frames ({frames / args.fps:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
