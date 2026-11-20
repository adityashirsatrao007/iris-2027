import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from torchvision import models, transforms
sys.path.insert(0, str(Path(__file__).parents[1]))
from src.common import seed_everything, save_json
from src.data import ImageManifest
from src.metrics import classification_metrics

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--val-fraction", type=float, default=.2)
p.add_argument("--size", type=int, default=224)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--workers", type=int, default=0)
a = p.parse_args()
seed_everything(a.seed)
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(a.manifest)
if df.source_id.nunique() < 2: raise ValueError("At least two source_id groups are required")
gss = GroupShuffleSplit(n_splits=1, test_size=a.val_fraction, random_state=a.seed)
train_i, val_i = next(gss.split(df, groups=df.source_id))
train_df, val_df = df.iloc[train_i], df.iloc[val_i]
normalize = transforms.Normalize([.485,.456,.406], [.229,.224,.225])
train_tf = transforms.Compose([transforms.Resize((a.size,a.size)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
val_tf = transforms.Compose([transforms.Resize((a.size,a.size)), transforms.ToTensor(), normalize])
train_dl = DataLoader(ImageManifest(train_df, train_tf), a.batch_size, shuffle=True, num_workers=a.workers, pin_memory=True)
val_dl = DataLoader(ImageManifest(val_df, val_tf), a.batch_size, shuffle=False, num_workers=a.workers, pin_memory=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 3); model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss(); best = -1.0
for epoch in range(1, a.epochs + 1):
    model.train(); total = 0.0
    for x, y, _ in train_dl:
        opt.zero_grad(); loss = loss_fn(model(x.to(device)), y.to(device)); loss.backward(); opt.step()
        total += loss.item() * len(y)
    model.eval(); ys = []; probs = []
    with torch.no_grad():
        for x, y, _ in val_dl:
            ys.extend(y.numpy()); probs.extend(torch.softmax(model(x.to(device)), 1).cpu().numpy())
    m = classification_metrics(np.asarray(ys), np.asarray(probs))
    print(f"epoch {epoch:03d} loss={total/len(train_df):.4f} val_macro_f1={m['macro_f1']:.4f}")
    if m['macro_f1'] > best:
        best = m['macro_f1']; torch.save(model.state_dict(), out / "best.pt"); save_json(m, out / "best_metrics.json")
save_json({"device": str(device), "epochs": a.epochs, "best_macro_f1": best,
           "train_rows": len(train_df), "val_rows": len(val_df)}, out / "run.json")

# Re-evaluate the best checkpoint and export predictions so that fine-tuned runs
# produce the same artifacts as the frozen-probe runs.
model.load_state_dict(torch.load(out / "best.pt", map_location=device))
model.eval()
ys, probs = [], []
with torch.no_grad():
    for x, y, _ in val_dl:
        ys.extend(y.numpy())
        probs.extend(torch.softmax(model(x.to(device)), 1).cpu().numpy())
y_true, y_prob = np.asarray(ys), np.asarray(probs)
save_json(classification_metrics(y_true, y_prob), out / "metrics.json")
pd.DataFrame({
    "path": val_df.path.values, "source_id": val_df.source_id.values, "true": y_true,
    "pred": y_prob.argmax(1),
    "p_original": y_prob[:, 0], "p_restored": y_prob[:, 1], "p_generated": y_prob[:, 2],
}).to_csv(out / "predictions.csv", index=False)
print(f"Wrote predictions.csv and metrics.json to {out}")
