from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def build_manifest(root, seed=42):
    rows = []
    for label in ("original", "restored", "generated"):
        directory = Path(root) / label
        for path in sorted(directory.rglob("*")) if directory.exists() else []:
            if path.suffix.lower() in EXTS:
                rows.append({"path": str(path.resolve()), "label": label,
                             "source_id": path.stem.split("__")[0],
                             "operator": path.parent.name if label == "restored" else "none"})
    if not rows: raise RuntimeError(f"No images found below {root}")
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

class ImageManifest(Dataset):
    def __init__(self, frame, transform=None):
        self.frame = frame.reset_index(drop=True); self.transform = transform
        self.labels = {x:i for i,x in enumerate(("original", "restored", "generated"))}
    def __len__(self): return len(self.frame)
    def __getitem__(self, i):
        row = self.frame.iloc[i]
        with Image.open(row.path) as im: image = im.convert("RGB")
        if self.transform: image = self.transform(image)
        return image, self.labels[row.label], row.path
