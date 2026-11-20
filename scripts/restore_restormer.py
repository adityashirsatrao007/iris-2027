"""Apply the official Restormer real-denoising model to a folder of images.

Restormer is a same-resolution restoration operator (scale 1), unlike
Real-ESRGAN which upsamples. Weights are the official `real_denoising.pth`
release mirrored at `deepinv/Restormer` on Hugging Face.

Architecture support comes from `spandrel_extra_arches`, so the Restormer
repository does not have to be vendored and `basicsr` is not required.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import spandrel
import spandrel_extra_arches

WEIGHTS = {
    "real-denoising": ("https://huggingface.co/deepinv/Restormer/resolve/main/real_denoising.pth",
                       "restormer_real_denoising.pth"),
    "gaussian-color-blind": ("https://huggingface.co/deepinv/Restormer/resolve/main/gaussian_color_denoising_blind.pth",
                             "restormer_gaussian_color_blind.pth"),
}

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--model", default="real-denoising", choices=sorted(WEIGHTS))
p.add_argument("--limit", type=int, default=0)
p.add_argument("--weights-dir", default="data/models")
p.add_argument("--threads", type=int, default=0, help="0 leaves torch defaults alone")
p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
               help="auto uses CUDA when a device is available")
p.add_argument("--overwrite", action="store_true",
               help="redo images whose output already exists (default: resume)")
a = p.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
device = torch.device("cuda" if (a.device == "auto" and torch.cuda.is_available()) else
                      ("cpu" if a.device == "auto" else a.device))

url, filename = WEIGHTS[a.model]
weights = Path(a.weights_dir) / filename
weights.parent.mkdir(parents=True, exist_ok=True)
if not weights.exists():
    import urllib.request
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, weights)

spandrel_extra_arches.install()
# Official checkpoints wrap the state dict in {'params': ...}
state = torch.load(str(weights), map_location="cpu", weights_only=False)
if isinstance(state, dict) and "params" in state:
    state = state["params"]
desc = spandrel.ModelLoader().load_from_state_dict(state)
model = desc.model.eval().to(device)
print(f"Loaded {filename} as {type(desc.architecture).__name__} on {device}", flush=True)

src = sorted(Path(a.input).glob("*.jpg")) + sorted(Path(a.input).glob("*.png"))
if a.limit:
    src = src[: a.limit]
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

PATTERN = 8  # Restormer downsamples 4 times, so H and W must be multiples of 8

started = time.time()
processed = 0
for path in src:
    target = out / f"{path.stem}__restormer.jpg"
    if not a.overwrite and target.exists() and target.stat().st_size > 0:
        continue
    image = Image.open(path).convert("RGB")
    w, h = image.size
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, _, th, tw = tensor.shape
    ph, pw = (-th) % PATTERN, (-tw) % PATTERN
    if ph or pw:
        tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        restored = model(tensor).clamp(0, 1)
    restored = restored[:, :, :th, :tw]
    restored = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray((restored * 255).round().astype(np.uint8))
    if result.size != (w, h):
        result = result.resize((w, h), Image.LANCZOS)
    result.save(target, quality=95)
    processed += 1
    if processed % 10 == 0 or processed == len(src):
        elapsed = time.time() - started
        print(f"{processed} new  {elapsed / processed:.2f}s/img  "
              f"eta {(len(src) - processed) * elapsed / processed / 60:.0f} min", flush=True)

present = len(list(out.glob("*.jpg")))
print(f"Restored {processed} new images into {out} with Restormer ({a.model}); "
      f"{present} outputs present")
