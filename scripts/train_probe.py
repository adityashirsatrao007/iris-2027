"""Train and evaluate a provenance classifier on frozen backbone features.

Feature extraction runs once, so repeated probes are cheap on CPU. Splitting is
grouped by source_id, which keeps an original and its restored version in the
same split and prevents leakage.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import seed_everything, save_json
from src.data import ImageManifest
from src.metrics import classification_metrics

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
p.add_argument("--backbone", default="resnet50")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--val-fraction", type=float, default=0.25)
p.add_argument("--max-iter", type=int, default=3000)
p.add_argument("--size", type=int, default=224,
               help="resize all images to this square size; use 128 to remove the resolution cue")
a = p.parse_args()

seed_everything(a.seed)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(a.manifest)

gss = GroupShuffleSplit(n_splits=1, test_size=a.val_fraction, random_state=a.seed)
train_i, val_i = next(gss.split(df, groups=df.source_id))
train_df = df.iloc[train_i].reset_index(drop=True)
val_df = df.iloc[val_i].reset_index(drop=True)

tf = transforms.Compose([
    transforms.Resize((a.size, a.size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval().to(device)


def embed(frame):
    dl = DataLoader(ImageManifest(frame, tf), a.batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    with torch.no_grad():
        for x, y, _ in dl:
            feats.append(model(x.to(device)).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


print(f"Extracting train features ({len(train_df)}) ...", flush=True)
Xtr, ytr = embed(train_df)
print(f"Extracting test features ({len(val_df)}) ...", flush=True)
Xte, yte = embed(val_df)

np.savez_compressed(out / "features.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)

clf = LogisticRegression(max_iter=a.max_iter, class_weight="balanced", n_jobs=-1)
clf.fit(Xtr, ytr)
joblib.dump(clf, out / "classifier.joblib")
probs = clf.predict_proba(Xte)
metrics = classification_metrics(yte, probs)
metrics.update({"backbone": a.backbone, "train_rows": len(train_df), "test_rows": len(val_df),
                "device": str(device), "seed": a.seed})
save_json(metrics, out / "metrics.json")

pd.DataFrame({
    "path": val_df.path, "source_id": val_df.source_id, "true": yte,
    "pred": probs.argmax(1),
    "p_original": probs[:, 0], "p_restored": probs[:, 1], "p_generated": probs[:, 2],
}).to_csv(out / "predictions.csv", index=False)

print(json.dumps({k: v for k, v in metrics.items() if k not in ("per_class", "confusion_matrix")}, indent=2))
