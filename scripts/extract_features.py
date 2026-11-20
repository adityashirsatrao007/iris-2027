"""Extract and cache backbone features for every image in the manifest."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.data import ImageManifest

LABELS = {"original": 0, "restored": 1, "generated": 2}
GENERATORS = ("biggan", "vqdm", "glide", "adm")

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True)
p.add_argument("--output", required=True)
p.add_argument("--size", type=int, default=128)
p.add_argument("--batch-size", type=int, default=32)
a = p.parse_args()

df = pd.read_csv(a.manifest)
df["y"] = df.label.map(LABELS)
df["generator"] = df.path.map(
    lambda p: next((g for g in GENERATORS if Path(p).name.startswith(g + "_")), "none"))

tf = transforms.Compose([
    transforms.Resize((a.size, a.size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Identity()
model.eval()

dl = DataLoader(ImageManifest(df, tf), a.batch_size, shuffle=False, num_workers=0)
feats = []
with torch.no_grad():
    for x, _, _ in dl:
        feats.append(model(x).numpy())

X = np.concatenate(feats)
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, X=X, y=df.y.values, path=df.path.values,
                    source_id=df.source_id.values, generator=df.generator.values)
print(f"Saved {X.shape} features to {out}")
