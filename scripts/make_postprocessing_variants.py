"""Create reproducible non-learned deployment-transformation variants."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/original")
    parser.add_argument("--output", default="data/raw/postprocessed")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    source = sorted(Path(args.input).glob("*.jpg"))
    if args.limit:
        source = source[: args.limit]
    for quality in (70, 50):
        out = Path(args.output) / f"jpeg{quality}"
        out.mkdir(parents=True, exist_ok=True)
        for path in source:
            image = Image.open(path).convert("RGB")
            image.save(out / f"{path.stem}__jpeg{quality}.jpg", quality=quality)

    out = Path(args.output) / "resize50"
    out.mkdir(parents=True, exist_ok=True)
    for path in source:
        image = Image.open(path).convert("RGB")
        small = image.resize(
            (max(1, image.width // 2), max(1, image.height // 2)),
            Image.Resampling.BICUBIC,
        )
        restored = small.resize(image.size, Image.Resampling.BICUBIC)
        restored.save(out / f"{path.stem}__resize50.jpg", quality=95)
    print(f"Created JPEG and resize variants for {len(source)} images in {args.output}")


if __name__ == "__main__":
    main()
