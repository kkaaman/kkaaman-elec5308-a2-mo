"""Compare frozen baseline/candidate on a held-out split and save overlays."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from train_lane import checkpoint_state, evaluate_model, make_dataset, model_class

from evaluation.dataset import local_file, read_dataset, save_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    import cv2
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    manifest, rows = read_dataset(args.dataset)
    rows = [r for r in rows if r["split"] == args.split]
    if not rows:
        raise ValueError("Requested evaluation split is empty")
    args.output.mkdir(parents=True, exist_ok=False)
    data = make_dataset(torch, cv2, args.dataset, rows, tuple(manifest["image_size_wh"]))
    loader = torch.utils.data.DataLoader(data, batch_size=4, num_workers=0)
    cls, source = model_class(args.repo)
    report = {
        "split": args.split,
        "frames": len(rows),
        "dataset_index_sha256": manifest["samples_sha256"],
        "architecture_sha256": sha256(source),
        "scope": "Road-marking pixels on held-out roads in Town10HD; not route following or unseen-town generalization",
        "models": {},
        "promotion": "NOT_DEPLOYED",
    }
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        model = cls(num_classes=2, pretrained=False).to(args.device)
        model.load_state_dict(checkpoint_state(torch, path, args.device), strict=True)
        result = evaluate_model(torch, model, loader, args.device)
        result["sha256"] = sha256(path)
        report["models"][name] = result
        with torch.inference_mode():
            for i in range(min(12, len(rows))):
                image, _ = data[i]
                mask = (
                    model(image.unsqueeze(0).to(args.device))
                    .argmax(1)[0]
                    .cpu()
                    .numpy()
                    .astype(bool)
                )
                rgb = np.asarray(
                    Image.open(local_file(args.dataset, rows[i]["rgb"])).convert("RGB")
                )
                overlay = rgb.copy()
                overlay[mask] = (0.4 * rgb[mask] + 0.6 * np.array([255, 40, 40])).astype(np.uint8)
                Image.fromarray(overlay).save(args.output / f"{i:03d}_{name}.png")
        del model
    save_json(args.output / "comparison.json", report)
    print(report)


if __name__ == "__main__":
    main()
