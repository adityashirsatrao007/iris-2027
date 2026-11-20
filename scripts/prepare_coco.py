"""Extract a reproducible subset of COCO val2017 without unzipping the archive."""
import argparse
import io
import random
import zipfile
from pathlib import Path

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True)
p.add_argument("--output", required=True)
p.add_argument("--limit", type=int, default=1000)
p.add_argument("--size", type=int, default=256, help="long side after resize; 0 keeps original")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

random.seed(a.seed)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(a.zip) as zf:
    names = sorted(n for n in zf.namelist() if n.lower().endswith(".jpg"))
    random.shuffle(names)
    selected = names[: a.limit]
    for n in selected:
        with zf.open(n) as fh:
            image = Image.open(io.BytesIO(fh.read())).convert("RGB")
        if a.size:
            image.thumbnail((a.size, a.size), Image.LANCZOS)
        image.save(out / Path(n).name, quality=95)

print(f"Wrote {len(selected)} images to {out}")
