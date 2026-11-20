"""Apply a pretrained Real-ESRGAN restoration and resample back to input size.

Weights are the official Real-ESRGAN releases. Inference goes through spandrel
so the fragile `basicsr` import is not required on modern torchvision.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader

WEIGHTS = {
    "general-x4v3": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
                     "realesr-general-x4v3.pth"),
    "x4plus": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
               "RealESRGAN_x4plus.pth"),
}

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--model", default="general-x4v3", choices=sorted(WEIGHTS))
p.add_argument("--limit", type=int, default=0)
p.add_argument("--weights-dir", default="data/weights")
p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
               help="auto uses CUDA when a device is available")
p.add_argument("--overwrite", action="store_true",
               help="redo images whose output already exists (default: resume)")
a = p.parse_args()

device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                      ("cpu" if a.device == "auto" else a.device))
print(f"Device: {device}", flush=True)

url, filename = WEIGHTS[a.model]
weights = Path(a.weights_dir) / filename
weights.parent.mkdir(parents=True, exist_ok=True)
if not weights.exists():
    import urllib.request
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, weights)

desc = ModelLoader().load_from_file(str(weights))
model = desc.model.eval().to(device)
scale = int(desc.scale)

src = sorted(Path(a.input).glob("*.jpg")) + sorted(Path(a.input).glob("*.png"))
if a.limit:
    src = src[: a.limit]
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

processed = 0
for path in src:
    target = out / f"{path.stem}__{a.model}.jpg"
    if not a.overwrite and target.exists() and target.stat().st_size > 0:
        continue
    image = Image.open(path).convert("RGB")
    w, h = image.size
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        restored = model(tensor).clamp(0, 1)
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray((restored * 255).round().astype(np.uint8))
    if result.size != (w, h):
        result = result.resize((w, h), Image.LANCZOS)
    result.save(target, quality=95)
    processed += 1
    if processed % 25 == 0:
        print(f"{processed} new / {len(src)} inputs", flush=True)

present = len(list(out.glob("*.jpg")))
print(f"Restored {processed} new images into {out} (scale x{scale}); {present} outputs present")
