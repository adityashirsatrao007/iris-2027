"""Generate the paper figures from saved predictions and protocol results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = ["original", "restored", "generated"]
COLORS = {"original": "#2E5E8C", "restored": "#C4622D", "generated": "#5B8C5A"}

p = argparse.ArgumentParser()
p.add_argument("--predictions", required=True)
p.add_argument("--protocols", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()

df = pd.read_csv(a.predictions)
df["true_name"] = df.true.map(dict(enumerate(LABELS)))
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. confusion matrix
cm = pd.crosstab(df.true_name, df.pred.map(dict(enumerate(LABELS)))).reindex(
    index=LABELS, columns=LABELS, fill_value=0)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
im = ax.imshow(cm.values, cmap="Blues")
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm.values[i, j]), ha="center", va="center",
                color="white" if cm.values[i, j] > cm.values.max() / 2 else "black")
ax.set_xticks(range(3), LABELS, rotation=20)
ax.set_yticks(range(3), LABELS)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Provenance confusion matrix")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(out / "fig_confusion_matrix.png", dpi=300)
plt.close(fig)

# 2. reliability diagram
probs = df[["p_original", "p_restored", "p_generated"]].values
conf, pred = probs.max(1), probs.argmax(1)
correct = (pred == df.true.values)
bins = np.linspace(0, 1, 11)
centers, accs, counts = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (conf > lo) & (conf <= hi)
    if mask.sum():
        centers.append((lo + hi) / 2)
        accs.append(correct[mask].mean())
        counts.append(int(mask.sum()))
fig, ax = plt.subplots(figsize=(4.0, 3.6))
ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
ax.plot(centers, accs, "o-", color="#2E5E8C", label="Model")
ax.set_xlabel("Confidence")
ax.set_ylabel("Accuracy")
ax.set_title("Reliability diagram")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "fig_reliability.png", dpi=300)
plt.close(fig)

# 3. false-generated rate vs threshold
thresholds = np.linspace(0.01, 0.99, 99)
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for name in ("original", "restored"):
    sub = df[df.true_name == name]
    rates = [(sub.p_generated >= t).mean() for t in thresholds]
    ax.plot(thresholds, rates, color=COLORS[name], label=name)
ax.axvline(0.5, ls=":", color="black", lw=1)
ax.text(0.51, ax.get_ylim()[1] * 0.9, "deployed\nthreshold", fontsize=7)
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Flagged as generated")
ax.set_title("False-generated rate on authentic images")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out / "fig_fpr_threshold.png", dpi=300)
plt.close(fig)

# 4. unseen-generator AUROC
protocols = json.loads(Path(a.protocols).read_text())
folds = protocols["leave_one_generator_out"]
families = list(folds)
aurocs = [folds[g]["auroc"] for g in families]
fig, ax = plt.subplots(figsize=(4.4, 3.4))
bars = ax.bar([g.upper() for g in families], aurocs, color="#2E5E8C")
ax.axhline(protocols["leave_one_generator_mean_auroc"], ls="--", color="#C4622D",
           label=f"mean = {protocols['leave_one_generator_mean_auroc']:.3f}")
for b, v in zip(bars, aurocs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("AUROC")
ax.set_title("Real vs generated with an unseen family")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(out / "fig_unseen_generator.png", dpi=300)
plt.close(fig)

print(f"Wrote 4 figures to {out}")
