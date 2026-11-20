"""Convert the COCO val2017 parquet mirror into individual JPEG files."""
import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--parquet", required=True)
p.add_argument("--output", required=True)
p.add_argument("--prefix", default="cocoval")
p.add_argument("--limit", type=int, default=0)
p.add_argument("--size", type=int, default=256, help="long side; 0 keeps native resolution")
a = p.parse_args()

table = pq.read_table(a.parquet, columns=["image"])
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

count = 0
for i, cell in enumerate(table.column("image").to_pylist()):
    if a.limit and count >= a.limit:
        break
    if cell is None:
        continue
    raw = cell["bytes"] if isinstance(cell, dict) else cell
    if not raw:
        continue
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    if a.size:
        image.thumbnail((a.size, a.size), Image.LANCZOS)
    image.save(out / f"{a.prefix}_{count:06d}.jpg", quality=95)
    count += 1

print(f"Wrote {count} images to {out}")
