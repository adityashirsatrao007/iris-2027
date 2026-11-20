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
