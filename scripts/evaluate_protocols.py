"""Evaluate the provenance classifier under several generalisation protocols.

Protocols
---------
1. three_class_random  : grouped random split, original/restored/generated.
2. leave_one_generator : binary original-vs-generated, one family fully unseen.

Everything runs on cached features, so the protocols are cheap to repeat.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

GENERATORS = ("biggan", "vqdm", "glide", "adm")

p = argparse.ArgumentParser()
p.add_argument("--features", required=True)
p.add_argument("--output", required=True)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

d = np.load(a.features, allow_pickle=True)
X, y = d["X"], d["y"]
source_id, generator = d["source_id"], d["generator"]
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
report = {}

# 1. three-class, grouped random split
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=a.seed)
tr, te = next(gss.split(X, y, groups=source_id))
clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
prob = clf.predict_proba(X[te])
pred = prob.argmax(1)
report["three_class_random"] = {
    "accuracy": float(accuracy_score(y[te], pred)),
    "macro_f1": float(f1_score(y[te], pred, average="macro")),
    "macro_auroc_ovr": float(roc_auc_score(y[te], prob, multi_class="ovr", average="macro")),
    "train_rows": int(len(tr)), "test_rows": int(len(te)),
}

# 2. binary original-vs-generated with one generator family unseen
real = y == 0
folds = {}
for g in GENERATORS:
    fake_g = y == 2
    is_g = generator == g
    # split the authentic images so the same photo never spans train/test
    real_idx = np.where(real)[0]
    real_groups = source_id[real_idx]
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=a.seed)
    real_tr, real_te = next(splitter.split(real_idx, groups=real_groups))
    train_idx = np.concatenate([real_idx[real_tr], np.where(fake_g & ~is_g)[0]])
    test_idx = np.concatenate([real_idx[real_te], np.where(fake_g & is_g)[0]])

    y_tr = (y[train_idx] == 2).astype(int)
    y_te = (y[test_idx] == 2).astype(int)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[train_idx], y_tr)
    score = clf.predict_proba(X[test_idx])[:, 1]
    folds[g] = {
        "accuracy": float(accuracy_score(y_te, (score >= 0.5).astype(int))),
        "auroc": float(roc_auc_score(y_te, score)),
        "n_test": int(len(test_idx)),
    }
report["leave_one_generator_out"] = folds
report["leave_one_generator_mean_auroc"] = float(np.mean([v["auroc"] for v in folds.values()]))

# 3. binary detector transfer: does a real-vs-fake classifier flag restored photos?
real_idx = np.where(y == 0)[0]
gen_idx = np.where(y == 2)[0]
rest_idx = np.where(y == 1)[0]

real_split = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=a.seed)
                  .split(real_idx, groups=source_id[real_idx]))
real_tr, real_te = real_idx[real_split[0]], real_idx[real_split[1]]
gen_split = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=a.seed)
                 .split(gen_idx, groups=source_id[gen_idx]))
gen_tr, gen_te = gen_idx[gen_split[0]], gen_idx[gen_split[1]]

train_idx = np.concatenate([real_tr, gen_tr])
y_tr = (y[train_idx] == 2).astype(int)
clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(X[train_idx], y_tr)

def fake_rate(idx):
    return float((clf.predict_proba(X[idx])[:, 1] >= 0.5).mean())

fpr_real = fake_rate(real_te)
fpr_rest = fake_rate(rest_idx)
report["binary_transfer"] = {
    "fpr_original": fpr_real,
    "fpr_restored": fpr_rest,
    "relative_increase": (fpr_rest / fpr_real) if fpr_real > 0 else None,
    "tpr_generated": fake_rate(gen_te),
    "auroc_original_vs_generated": float(roc_auc_score(
        np.r_[np.zeros(len(real_te)), np.ones(len(gen_te))],
        np.r_[clf.predict_proba(X[real_te])[:, 1], clf.predict_proba(X[gen_te])[:, 1]])),
    "n": {"original_test": int(len(real_te)), "restored": int(len(rest_idx)), "generated_test": int(len(gen_te))},
}

(out / "protocols.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

lines = ["# Generalisation protocols", ""]
lines.append("## Three-class, grouped random split")
lines.append("")
r = report["three_class_random"]
lines.append(f"- accuracy: {r['accuracy']*100:.2f}%")
lines.append(f"- macro-F1: {r['macro_f1']*100:.2f}%")
lines.append(f"- macro AUROC: {r['macro_auroc_ovr']:.3f}")
lines.append("")
lines.append("## Real vs generated, one generator family unseen")
lines.append("")
lines.append("| Unseen family | n | Accuracy | AUROC |")
lines.append("|---|---|---|---|")
for g, v in folds.items():
    lines.append(f"| {g} | {v['n_test']} | {v['accuracy']*100:.2f}% | {v['auroc']:.3f} |")
lines.append("")
lines.append(f"Mean unseen-family AUROC: **{report['leave_one_generator_mean_auroc']:.3f}**")
lines.append("")
bt = report["binary_transfer"]
lines.append("## Binary detector transfer (the abstract's claim)")
lines.append("")
lines.append("A real-vs-fake classifier trained only on `original` and `generated`, then")
lines.append("applied to unseen authentic photographs and to their restored versions.")
lines.append("")
lines.append(f"- FPR on pristine authentic: {bt['fpr_original']*100:.2f}%")
lines.append(f"- FPR on restored authentic: {bt['fpr_restored']*100:.2f}%")
if bt["relative_increase"] is not None:
    lines.append(f"- relative increase: **{bt['relative_increase']:.1f}x**")
lines.append(f"- TPR on held-out generated: {bt['tpr_generated']*100:.2f}%")
lines.append(f"- AUROC original vs generated: {bt['auroc_original_vs_generated']:.3f}")
(out / "PROTOCOLS.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
