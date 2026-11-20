"""Build paper-ready tables and a false-positive-rate report from predictions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

LABELS = {0: "original", 1: "restored", 2: "generated"}


def bootstrap_ci(values, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    stats = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


p = argparse.ArgumentParser()
p.add_argument("--predictions", required=True)
p.add_argument("--output", required=True)
p.add_argument("--threshold", type=float, default=0.5)
a = p.parse_args()

df = pd.read_csv(a.predictions)
df["true_name"] = df.true.map(LABELS)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

lines = ["# Provenance model report", ""]
lines.append(f"Rows: {len(df)}  |  decision threshold: {a.threshold}")
lines.append("")

lines.append("## Per-class detection")
lines.append("")
lines.append("| True class | n | Correct | Recall | 95% CI |")
lines.append("|---|---|---|---|---|")
for name in ("original", "restored", "generated"):
    sub = df[df.true_name == name]
    if len(sub) == 0:
        continue
    correct = (sub.pred == sub.true).astype(int).values
    lo, hi = bootstrap_ci(correct)
    lines.append(f"| {name} | {len(sub)} | {correct.sum()} | {correct.mean()*100:.2f}% | "
                 f"[{lo*100:.2f}, {hi*100:.2f}] |")
lines.append("")

lines.append("## False-generated rate (paper's FPR analogue)")
lines.append("")
lines.append("Share of authentic images whose `p_generated` exceeds the threshold.")
lines.append("")
lines.append("| Condition | n | Flagged as generated | 95% CI |")
lines.append("|---|---|---|---|")
for name in ("original", "restored"):
    sub = df[df.true_name == name]
    if len(sub) == 0:
        continue
    flagged = (sub.p_generated >= a.threshold).astype(int).values
    lo, hi = bootstrap_ci(flagged)
    lines.append(f"| {name} | {len(sub)} | {flagged.mean()*100:.2f}% | "
                 f"[{lo*100:.2f}, {hi*100:.2f}] |")
lines.append("")

orig = df[df.true_name == "original"]
rest = df[df.true_name == "restored"]
if len(orig) and len(rest):
    fpr_o = (orig.p_generated >= a.threshold).mean()
    fpr_r = (rest.p_generated >= a.threshold).mean()
    if fpr_o > 0:
        lines.append(f"Relative increase in false-generated rate after restoration: "
                     f"**{fpr_r / fpr_o:.1f}x**")
    else:
        lines.append("Relative increase could not be computed because the pristine rate was zero.")
    lines.append("")

lines.append("## Confusion matrix (rows = true, cols = predicted)")
lines.append("")
cm = pd.crosstab(df.true_name, df.pred.map(LABELS))
lines.append(cm.to_markdown())
lines.append("")

# Binary contrasts isolate provenance from content: original-vs-restored shares
# the same source photographs, so any separation is caused by the restorer.
from sklearn.metrics import roc_auc_score

def binary_block(name, a_class, b_class, score_col, positive_is_b=True):
    sub = df[df.true_name.isin([a_class, b_class])]
    if len(sub) == 0 or sub.true_name.nunique() < 2:
        return
    y = (sub.true_name == b_class).astype(int).values
    s = sub[score_col].values if score_col in sub else None
    if s is None:
        return
    auc = roc_auc_score(y, s) if positive_is_b else roc_auc_score(y, -s)
    lines.append(f"- **{name}**: n={len(sub)}, AUROC={auc:.3f}")
    lines.append("")

lines.append("## Binary contrasts")
lines.append("")
binary_block("original vs generated (different content sources)", "original", "generated", "p_generated")
binary_block("original vs restored (same source photographs)", "original", "restored", "p_restored")


(out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
