"""Create publication metrics and figures from saved predictions.

No model is retrained here. Every statistic is computed from saved per-image
predictions, with paired original/restored comparisons where applicable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import auc, roc_curve, roc_auc_score


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, n=5000):
    samples = rng.choice(values, (n, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def detector_stats(path: str, seed: int):
    df = pd.read_csv(path)
    original = df[df.condition == "original"].sort_values("path")
    restored = df[df.condition == "restored"].sort_values("path")
    if len(original) != len(restored):
        raise ValueError(f"unpaired detector file: {path}")
    delta = restored.synthetic_probability.to_numpy() - original.synthetic_probability.to_numpy()
    rng = np.random.default_rng(seed)
    out = {
        "n": int(len(delta)),
        "mean_original": float(original.synthetic_probability.mean()),
        "mean_restored": float(restored.synthetic_probability.mean()),
        "mean_delta": float(delta.mean()),
        "mean_delta_bootstrap_95": bootstrap_mean(delta, rng),
        "median_delta": float(np.median(delta)),
        "increased": int((delta > 0).sum()),
        "decreased": int((delta < 0).sum()),
        "unchanged": int((delta == 0).sum()),
        "thresholds": {},
    }
    try:
        stat = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        out["wilcoxon_p"] = float(stat.pvalue)
    except ValueError:
        out["wilcoxon_p"] = None
    for threshold in (0.01, 0.05, 0.10, 0.50):
        a = original.synthetic_probability.to_numpy() >= threshold
        b = restored.synthetic_probability.to_numpy() >= threshold
        out["thresholds"][str(threshold)] = {
            "original_rate": float(a.mean()),
            "restored_rate": float(b.mean()),
            "paired_increase": int((~a & b).sum()),
            "paired_decrease": int((a & ~b).sum()),
        }
    return out, original, restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="results/demo_probe_224/predictions.csv")
    parser.add_argument("--output", default="results/publication_metrics.json")
    parser.add_argument("--figures", default="paper/figures")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--detectors", nargs="+", default=[
        "results/cnnspot_official.csv",
        "results/cnnspot_official_restormer.csv",
        "results/cnnspot_official_prob01_realesr.csv",
        "results/cnnspot_official_prob01_restormer.csv",
        "results/univfd_official.csv",
        "results/univfd_official_restorer.csv",
        "results/cnnspot_official_jpeg70.csv",
        "results/cnnspot_official_jpeg50.csv",
        "results/cnnspot_official_resize50.csv",
        "results/univfd_official_jpeg70.csv",
        "results/univfd_official_jpeg50.csv",
        "results/cnnspot_official_genimage_same_benchmark.csv",
        "results/univfd_official_genimage_same_benchmark.csv",
    ])
    args = parser.parse_args()
    out = Path(args.output)
    figdir = Path(args.figures)
    figdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {"seed": args.seed, "detectors": {}}

    for path in args.detectors:
        stats, original, restored = detector_stats(path, args.seed)
        report["detectors"][str(Path(path))] = stats

    # Probe one-vs-rest ROC curves.
    pred = pd.read_csv(args.predictions)
    probs = pred[["p_original", "p_restored", "p_generated"]].to_numpy()
    y = pred.true.to_numpy()
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    colors = ["#2E5E8C", "#C4622D", "#5B8C5A"]
    labels = ["Original", "Restored", "Generated"]
    for i, (label, color) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve((y == i).astype(int), probs[:, i])
        score = roc_auc_score((y == i).astype(int), probs[:, i])
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUROC {score:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Probe one-vs-rest ROC curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "fig_probe_roc.png", dpi=300); plt.close(fig)

    # Detector mean paired changes with bootstrap intervals.
    selected = [p for p in args.detectors if "genimage_same" not in p and "jpeg" not in p and "resize" not in p]
    names, means, lows, highs = [], [], [], []
    for path in selected:
        s = report["detectors"][str(Path(path))]
        names.append(Path(path).stem.replace("cnnspot_official_", "CNNSpot ").replace("univfd_official_", "UnivFD "))
        means.append(s["mean_delta"]); lows.append(s["mean_delta"] - s["mean_delta_bootstrap_95"][0]); highs.append(s["mean_delta_bootstrap_95"][1] - s["mean_delta"])
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    yloc = np.arange(len(names))
    ax.errorbar(means, yloc, xerr=[lows, highs], fmt="o", color="#2E5E8C", capsize=3)
    ax.axvline(0, color="0.4", ls="--", lw=1)
    ax.set_yticks(yloc, names, fontsize=8)
    ax.set_xlabel("Restored minus original mean synthetic probability")
    ax.set_title("Paired detector score changes (95% bootstrap intervals)")
    fig.tight_layout(); fig.savefig(figdir / "fig_detector_delta.png", dpi=300); plt.close(fig)

    # Threshold curves for the six primary learned-detector/restorer conditions.
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    thresholds = np.linspace(0.01, 0.99, 99)
    for path in selected:
        _, original, restored = detector_stats(path, args.seed)
        label = Path(path).stem.replace("cnnspot_official_", "CNNSpot ").replace("univfd_official_", "UnivFD ")
        ax.plot(thresholds, [(restored.synthetic_probability >= t).mean() - (original.synthetic_probability >= t).mean() for t in thresholds], label=label)
    ax.axhline(0, color="0.4", ls="--", lw=1)
    ax.set(xlabel="Decision threshold", ylabel="Change in flagged rate", title="Restoration-induced detector-rate change")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(figdir / "fig_detector_threshold_delta.png", dpi=300); plt.close(fig)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out} and figures to {figdir}")


if __name__ == "__main__":
    main()
