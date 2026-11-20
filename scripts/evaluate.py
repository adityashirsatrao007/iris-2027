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
from src.metrics import classification_metrics
from src.common import save_json

p = argparse.ArgumentParser()
p.add_argument("--manifest", required=True); p.add_argument("--output", required=True)
p.add_argument("--weights"); p.add_argument("--batch-size", type=int, default=32)
a = p.parse_args()
df = pd.read_csv(a.manifest)
tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
dl = DataLoader(ImageManifest(df, tf), a.batch_size, shuffle=False, num_workers=0)
model = models.resnet50(weights=None); model.fc = torch.nn.Linear(model.fc.in_features, 3)
if a.weights: model.load_state_dict(torch.load(a.weights, map_location="cpu"))
model.eval(); ys = []; probs = []; paths = []
with torch.no_grad():
    for x, y, pths in dl:
        ys.extend(y.numpy()); probs.extend(torch.softmax(model(x), 1).numpy()); paths.extend(pths)
probs = np.asarray(probs); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
save_json(classification_metrics(np.asarray(ys), probs), out / "metrics.json")
pd.DataFrame({"path": paths, "true": ys, "pred": probs.argmax(1), "p_original": probs[:,0], "p_restored": probs[:,1], "p_generated": probs[:,2]}).to_csv(out / "predictions.csv", index=False)
print((out / "metrics.json").read_text())
