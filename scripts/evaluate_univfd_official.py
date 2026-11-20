"""Evaluate the released UnivFD checkpoint on paired authentic images.

The model and preprocessing are from the authors' UniversalFakeDetect code:
CLIP ViT-L/14 with its native resize/center-crop transform and CLIP
normalization. The checkpoint is not retrained here.
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

ROOT = Path(__file__).resolve().parents[1]
UNIVFD_DIR = ROOT / "external" / "UniversalFakeDetect"
sys.path.insert(0, str(UNIVFD_DIR))
from models import get_model  # noqa: E402


class PairedImages(Dataset):
    def __init__(self, original: Path, restored: Path, suffix: str, preprocess):
        originals = sorted(original.glob("*.jpg"))
        pairs = [
            (src, restored / f"{src.stem}{suffix}")
            for src in originals
            if (restored / f"{src.stem}{suffix}").exists()
        ]
        if not pairs:
            raise RuntimeError("no original/restored pairs found")
        self.rows = [(p[0], "original") for p in pairs] + [
            (p[1], "restored") for p in pairs
        ]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, condition = self.rows[index]
        return self.preprocess(Image.open(path).convert("RGB")), str(path), condition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="data/official_models/univfd_fc_weights.pth",
    )
    ap.add_argument("--restored-dir", default="data/raw/restored")
    ap.add_argument("--restored-suffix", default="__general-x4v3.jpg")
    ap.add_argument("--original-dir", default="data/raw/original")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output", default="results/univfd_official.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model("CLIP:ViT-L/14")
    state = torch.load(args.model, map_location="cpu")
    model.fc.load_state_dict(state)
    model.eval().to(device)
    data = PairedImages(
        ROOT / args.original_dir,
        ROOT / args.restored_dir,
        args.restored_suffix,
        model.preprocess,
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
        for batch_index, (images, paths, conditions) in enumerate(loader, 1):
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
            if batch_index == 1 or batch_index % 25 == 0:
                print(
                    f"processed {min(batch_index * args.batch_size, len(data))}/"
                    f"{len(data)} images",
                    flush=True,
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
        "model": "official UnivFD CLIP:ViT-L/14",
        "device": str(device),
        "summary": summary,
    }
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
