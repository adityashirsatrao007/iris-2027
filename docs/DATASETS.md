# Data provenance

Record the exact values below in the paper's data statement and in the artifact
metadata. Do not redistribute images whose license forbids it.

## Authentic photographs

| Field | Value |
|---|---|
| Source | COCO val2017 (Lin et al., *Microsoft COCO*, ECCV 2014) |
| URL | https://cocodataset.org/#download |
| Mirror used | `simopippa/reduced_coco_1000_val2017` on Hugging Face |
| Sampled | the full mirror (1,000 of the 5,000 val2017 images) |
| Preprocessing | stored at native resolution, JPEG quality 95 |
| License | COCO annotations CC BY 4.0; images from Flickr under their own licenses |

## Generated images

All four subsets are per-generator exports of the GenImage benchmark
(Zhu et al., *GenImage*, NeurIPS 2023 Datasets and Benchmarks).

| Subset | Generator family | Rows taken |
|---|---|---|
| `bitmind/GenImage_BigGAN` | BigGAN | 250 |
| `bitmind/GenImage_VQDM` | VQDM | 250 |
| `bitmind/GenImage_ADM` | ADM | 250 |
| `bitmind/GenImage_glide` | GLIDE | 250 |

Images are 128x128. GenImage's own real class is derived from ImageNet; this
project instead uses COCO val2017 as the authentic class, so the real and
generated sets are **not** drawn from the same source distribution. State this
as a limitation and, for the camera-ready version, repeat the experiment with
GenImage's matched real split.

## Restored images

Produced locally with Real-ESRGAN (`realesr-general-x4v3`, x4) from the
authentic set, then Lanczos-resampled back to the input dimensions. Nothing
else is changed. File names follow `{source_stem}__{model}.jpg` so the manifest
keeps the restored image grouped with its source photograph.

Citation:

> Xintao Wang et al. *Real-ESRGAN: Training Real-World Blind Super-Resolution
> with Pure Synthetic Data.* ICCV Workshops, 2021.

## Duplicate and leakage control

- Group every split by `source_id`; an original and its restored copy always
  land in the same split.
- Keep at least one generator family out of training for the cross-generator
  result.
- Record the dataset revision or export date next to every result.

## Same-benchmark control extension

The repository also provides `scripts/prepare_genimage_matched.py`. It streams
the `nebula/GenImage-arrow` `biggan-test` split and exports an equal number of
real and generated images from the same benchmark. The public Arrow release
does not expose a reliable pixel-pair identifier, so this is a
same-benchmark/distribution control, **not an exact source-paired dataset**.
It checks whether replacing COCO real images changes the result, but it does
not remove every possible content or class confound.

Example:

```powershell
python scripts/prepare_genimage_matched.py `
  --pairs 500 `
  --output data/matched_genimage
```

This extension is derived from the GenImage release and is subject to its
CC-BY-NC-SA-4.0 and dataset-specific non-commercial terms. Do not commit the
exported images to Git.
