"""Summarize paired detector CSVs with operating curves and bootstrap intervals.

The official evaluators write one row per image and condition. This script adds
uncertainty and threshold sensitivity without changing any detector scores.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, centre - half)), float(min(1, centre + half))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--output", default="results/detector_audit_summary.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    report = {"thresholds": [0.01, 0.05, 0.10, 0.50], "files": {}}

    for filename in args.csv:
        frame = pd.read_csv(filename)
        originals = frame[frame.condition == "original"].sort_values("path")
        restored = frame[frame.condition == "restored"].sort_values("path")
        a = originals.synthetic_probability.to_numpy(float)
        b = restored.synthetic_probability.to_numpy(float)
        if len(a) != len(b) or len(a) == 0:
            raise ValueError(f"unpaired or empty CSV: {filename}")
        thresholds = {}
        for threshold in report["thresholds"]:
            k_a = int((a >= threshold).sum())
            k_b = int((b >= threshold).sum())
            thresholds[str(threshold)] = {
                "original_rate": float(k_a / len(a)),
                "original_wilson_95": wilson(k_a, len(a)),
                "restored_rate": float(k_b / len(b)),
                "restored_wilson_95": wilson(k_b, len(b)),
            }
        delta = b - a
        boot = rng.choice(delta, size=(args.bootstrap, len(delta)), replace=True).mean(axis=1)
        report["files"][str(Path(filename))] = {
            "n": int(len(a)),
            "mean_original": float(a.mean()),
            "mean_restored": float(b.mean()),
            "mean_delta": float(delta.mean()),
            "mean_delta_bootstrap_95": [
                float(np.quantile(boot, 0.025)),
                float(np.quantile(boot, 0.975)),
            ],
            "increased": int((delta > 0).sum()),
            "decreased": int((delta < 0).sum()),
            "unchanged": int((delta == 0).sum()),
            "thresholds": thresholds,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
