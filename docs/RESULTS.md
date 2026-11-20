# Results (pilot run, 20 Nov 2026)

All numbers were produced by the scripts in this repository on this machine.
Nothing is copied from the abstract.

## Setup

- Backbone: ImageNet ResNet-50, frozen, 2048-d penultimate features.
- Head: multinomial logistic regression, `class_weight="balanced"`.
- Split: grouped by `source_id` (an original and its restored copy never span
  the split), 75/25 train/test, seed 42.
- Data: 1,000 COCO val2017 + 1,000 Real-ESRGAN restorations + 1,000 GenImage.
- Hardware: CPU only (AMD64, no CUDA).

## Three-class provenance classification

| Input size | Accuracy | Balanced acc. | Macro-F1 | Macro AUROC | ECE |
|---|---|---|---|---|---|
| 128 px (matched to GenImage) | 82.29% | 81.62% | 81.62% | 0.926 | 0.141 |
| 224 px | **88.28%** | 87.78% | 87.77% | **0.957** | 0.073 |

Per class at 224 px:

| Class | Recall | 95% CI |
|---|---|---|
| original | 81.62% | [76.50, 86.32] |
| restored | 82.48% | [77.35, 86.75] |
| generated | 99.25% | [98.12, 100.00] |

The confusion is almost entirely between `original` and `restored`
(40 + 39 of 468). `generated` is nearly separable, but that number is inflated
by the content confound described below.

## Binary contrasts (224 px)

| Contrast | AUROC |
|---|---|
| original vs generated | 0.998 |
| original vs restored (same photographs) | 0.889 |

Restoration is detectable from its own source photograph, but not perfectly.

## Unseen generator family (128 px, binary real vs generated)

| Held-out family | Accuracy | AUROC |
|---|---|---|
| BigGAN | 98.27% | 0.999 |
| VQDM | 98.40% | 0.998 |
| GLIDE | 80.27% | 0.932 |
| ADM | 88.00% | 0.982 |

Mean unseen-family AUROC: **0.978**. GLIDE is clearly the hardest transfer.

## Official detector audit

The released CNNSpot and UnivFD checkpoints were subsequently evaluated on the
same 1,000 COCO originals and their 1,000 restorations. Preprocessing followed
the released code: a 224-pixel center crop and ImageNet normalization for CNNSpot,
and the native CLIP ViT-L/14 transform for UnivFD. Values are rates at score 0.5.

| Detector/checkpoint | Restorer | Original | Restored |
|---|---|---:|---:|
| CNNSpot Blur+JPEG(0.5) | Real-ESRGAN | 0.30% | 0.20% |
| CNNSpot Blur+JPEG(0.5) | Restormer | 0.30% | 0.30% |
| CNNSpot Blur+JPEG(0.1) | Real-ESRGAN | 1.20% | 0.90% |
| CNNSpot Blur+JPEG(0.1) | Restormer | 1.20% | 1.70% |
| UnivFD CLIP ViT-L/14 | Real-ESRGAN | 0.60% | 1.60% |
| UnivFD CLIP ViT-L/14 | Restormer | 0.60% | 1.60% |

These official runs do not reproduce the abstract's 15.90% super-resolution or
5.20% denoising rates. UnivFD rises from 0.60% to 1.60% for both restorers;
CNNSpot's two released variants are detector- and restorer-dependent.

## Does restoration get pushed into the generated class?

A real-vs-fake classifier trained only on `original` and `generated`, then
applied to unseen authentic images and their restorations:

| Condition | Flagged as generated |
|---|---|
| pristine authentic | 1.00% |
| restored authentic | 0.30% |

**The abstract's 0.5% → 15.9% finding is not reproduced.** With a generic
ResNet-50 probe, restoration moved authentic images *away* from the generated
class. The official audit above measures the released CNNSpot and UnivFD
checkpoints directly and still does not produce the abstract's rates.

## Threats to validity

1. **Content confound.** Authentic images are COCO photographs; generated images
   are GenImage (ImageNet-derived). A classifier can exploit scene statistics
   rather than synthesis artefacts. This inflates the `generated` recall and the
   original-vs-generated AUROC. The fix is GenImage's matched real split, or a
   second real source drawn from the same distribution.
2. **Two restorers, two families.** Real-ESRGAN and Restormer were run, but each
   represents only one super-resolution or denoising operator; no broad
   leave-one-restorer-out result exists.
3. **Resolution.** GenImage is 128 px, so the primary comparison is at 128 px.
   The 224 px gain partly reflects upsampling, not extra evidence.
4. **Checkpoint and preprocessing scope.** Official CNNSpot and UnivFD
   checkpoints were run, but exact checkpoint variants and preprocessing choices
   affect the result; the abstract does not specify enough detail to guarantee an
   exact reproduction.
5. **Single seed.** No variance across seeds is reported yet.
