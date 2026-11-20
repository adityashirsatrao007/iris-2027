"""Export a balanced real/fake subset from a GenImage Arrow split.

The released Arrow data exposes real and generated rows, but not a reliable
pixel-pair identifier. This script therefore creates a same-benchmark,
balanced control set without claiming exact source pairing.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nebula/GenImage-arrow")
    parser.add_argument("--config", default="biggan-test")
    parser.add_argument("--split", default="test")
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--output", default="data/matched_genimage")
    args = parser.parse_args()

    output = Path(args.output)
    real_dir = output / "original"
    fake_dir = output / "generated"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset, args.config, split=args.split, streaming=True
    )
    counts = {0: 0, 1: 0}
    rows = []

    for row in dataset:
        label = int(row["label"])
        if label not in (0, 1):
            continue
        if counts[label] >= args.pairs:
            continue

        source_key = Path(str(row["image_path"])).name
        safe_key = Path(source_key).stem.replace(" ", "_")
        prefix = "real" if label == 0 else "fake"
        target_dir = real_dir if label == 0 else fake_dir
        target = target_dir / f"genimage_{prefix}_{counts[label]:06d}_{safe_key}.jpg"
        image = row["image"]
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        image.convert("RGB").save(target, quality=95)
        rows.append(
            {
                "path": str(target),
                "label": "original" if label == 0 else "generated",
                "source_id": f"genimage_{prefix}_{counts[label]:06d}",
                "generator": "matched-real" if label == 0 else args.config,
                "source_image": source_key,
            }
        )
        counts[label] += 1
        if counts[0] >= args.pairs and counts[1] >= args.pairs:
            break

    if counts[0] < args.pairs or counts[1] < args.pairs:
        raise RuntimeError(
            f"Found real={counts[0]}, generated={counts[1]}; "
            f"requested {args.pairs} of each."
        )
    (output / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    print(
        f"Exported a balanced same-benchmark subset: "
        f"{counts[0]} real and {counts[1]} generated images to {output}"
    )


if __name__ == "__main__":
    main()
