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

## Does restoration get pushed into the generated class?

A real-vs-fake classifier trained only on `original` and `generated`, then
applied to unseen authentic images and their restorations:

| Condition | Flagged as generated |
|---|---|
| pristine authentic | 1.00% |
| restored authentic | 0.30% |

**This does not reproduce the abstract's 0.5% → 15.9% finding.** With a generic
ResNet-50 probe, restoration moved authentic images *away* from the generated
class. The abstract's numbers come from CNNSpot and UnivFD, which were **not
executed here**. Until those official checkpoints are run on the same images,
the paper cannot claim the 15.9% or 5.2% figures.

## Threats to validity

1. **Content confound.** Authentic images are COCO photographs; generated images
   are GenImage (ImageNet-derived). A classifier can exploit scene statistics
   rather than synthesis artefacts. This inflates the `generated` recall and the
   original-vs-generated AUROC. The fix is GenImage's matched real split, or a
   second real source drawn from the same distribution.
2. **One restorer.** Only Real-ESRGAN was run. Restormer is untested, so no
   leave-one-restorer-out result exists.
3. **Resolution.** GenImage is 128 px, so the primary comparison is at 128 px.
   The 224 px gain partly reflects upsampling, not extra evidence.
4. **No official detectors.** CNNSpot and UnivFD are the paper's core claim and
   are missing.
5. **Single seed.** No variance across seeds is reported yet.
