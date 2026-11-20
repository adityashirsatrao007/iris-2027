"""Export images from a Hugging Face dataset into the project layout.

Streaming mode is the default so multi-gigabyte datasets are not downloaded in
full. Every exported file is named after the dataset and row index, which makes
the manifest's source_id unique per generated image.
"""
import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset

p = argparse.ArgumentParser()
p.add_argument("--dataset", required=True)
p.add_argument("--config", default=None)
p.add_argument("--split", default="train")
p.add_argument("--image-column", default="image")
p.add_argument("--output", required=True)
p.add_argument("--revision", default=None)
p.add_argument("--limit", type=int, default=250)
p.add_argument("--offset", type=int, default=0)
p.add_argument("--streaming", action="store_true", default=True)
p.add_argument("--no-streaming", dest="streaming", action="store_false")
p.add_argument("--prefix", default=None)
a = p.parse_args()

kwargs = {"split": a.split, "streaming": a.streaming}
if a.config:
    kwargs["name"] = a.config
if a.revision:
    kwargs["revision"] = a.revision
ds = load_dataset(a.dataset, **kwargs)

if a.offset:
    ds = ds.skip(a.offset)
ds = ds.take(a.limit)

slug = a.prefix or re.sub(r"[^A-Za-z0-9]+", "_", a.dataset).strip("_")
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)

count = 0
with (out / f"metadata_{slug}.jsonl").open("w", encoding="utf-8") as f:
    for i, row in enumerate(ds):
        image = row[a.image_column]
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        name = f"{slug}_{a.offset + i:06d}"
        path = out / f"{name}.jpg"
        image.save(path, quality=95)
        f.write(json.dumps({
            "path": str(path), "dataset": a.dataset, "revision": a.revision,
            "split": a.split, "row": a.offset + i, "source_id": name,
        }) + "\n")
        count += 1

print(f"Exported {count} images to {out}")
