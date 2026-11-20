"""Evaluate the released CNNSpot checkpoint on paired authentic images.

This follows the released CNNDetection preprocessing: ImageNet normalization and
a 224-pixel center crop, without a resize because the input corpus is already
prepared at the detector's native scale. It reports the 0.5 threshold rates
before/after Real-ESRGAN restoration and writes all per-image scores.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CNN_DIR = ROOT / "external" / "CNNDetection"
sys.path.insert(0, str(CNN_DIR))
from networks.resnet import resnet50  # noqa: E402


class PairedImages(Dataset):
    def __init__(self, original: Path, restored: Path, suffix: str):
        originals = sorted(original.glob("*.jpg"))
        pairs = []
        for src in originals:
            dst = restored / f"{src.stem}{suffix}"
            if dst.exists():
                pairs.append((src, dst))
        if not pairs:
            raise RuntimeError("no original/restored pairs found")
        self.rows = [(p[0], "original") for p in pairs] + [
            (p[1], "restored") for p in pairs
        ]
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, condition = self.rows[index]
        return self.transform(Image.open(path).convert("RGB")), str(path), condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="data/official_models/cnnspot_blur_jpg_prob0.5.pth",
    )
    ap.add_argument("--restored-dir", default="data/raw/restored")
    ap.add_argument("--restored-suffix", default="__general-x4v3.jpg")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output", default="results/cnnspot_official.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(num_classes=1)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval().to(device)
    data = PairedImages(
        ROOT / "data/raw/original",
        ROOT / args.restored_dir,
        args.restored_suffix,
    )
    loader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    rows = []
    with torch.inference_mode():
        for images, paths, conditions in loader:
            scores = model(images.to(device)).sigmoid().flatten().cpu().numpy()
            rows.extend(
                {
                    "path": path,
                    "condition": condition,
                    "synthetic_probability": float(score),
                    "flagged_at_0_5": int(score >= 0.5),
                }
                for path, condition, score in zip(paths, conditions, scores)
            )

    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for condition in ("original", "restored"):
        vals = [r for r in rows if r["condition"] == condition]
        summary[condition] = {
            "n": len(vals),
            "flagged_at_0_5": sum(r["flagged_at_0_5"] for r in vals),
            "false_positive_rate": sum(r["flagged_at_0_5"] for r in vals)
            / len(vals),
            "mean_score": float(np.mean([r["synthetic_probability"] for r in vals])),
        }
    report = {
        "checkpoint": str(Path(args.model)),
        "preprocessing": "CenterCrop(224), ImageNet normalization, no resize",
        "device": str(device),
        "summary": summary,
    }
    (out.with_suffix(".json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
