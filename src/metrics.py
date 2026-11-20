import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

def expected_calibration_error(y, probs, bins=10):
    conf = probs.max(1); pred = probs.argmax(1); ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0, 1, bins+1)[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any(): ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(ece)

def classification_metrics(y, probs):
    pred = probs.argmax(1)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro")),
           "ece": expected_calibration_error(y, probs),
           "confusion_matrix": confusion_matrix(y, pred).tolist(),
           "per_class": classification_report(y, pred, output_dict=True, zero_division=0)}
    try: out["macro_auroc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro"))
    except ValueError: out["macro_auroc_ovr"] = None
    return out
