# Experiment protocol

## Dataset matrix

| Class | Minimum sources | Requirement |
|---|---|---|
| Original | COCO val2017 plus a second authentic source | source image appears in one split only |
| Restored | Restormer, Real-ESRGAN, one low-light/deblur method | hold out one operator |
| Generated | four generator families | hold out one family |

For every derived image retain `source_id`, `operator`, `operator_version`, and `seed`. Never split after making random crops.

## Required reporting

Run pristine-vs-generated, before/after restoration, three-class classification, cross-operator, cross-generator, calibration, and frequency/augmentation ablations. Report accuracy, balanced accuracy, macro-F1, per-class precision/recall, confusion matrix, one-vs-rest AUROC, ECE, FPR/TPR at fixed thresholds, and 95% bootstrap confidence intervals. Use Wilcoxon signed-rank testing for paired detector score changes.

For official detector CSVs, use `scripts/summarize_detector_audit.py`. It
reports Wilson intervals for FPR at 1%, 5%, 10%, and 50%, plus a paired
bootstrap interval for the mean score change. This prevents a single
threshold from hiding operating-point dependence.

For the paper-ready detector statistics and figures, run:

```powershell
python scripts/make_publication_metrics.py
```

This writes `results/publication_metrics.json` and produces one-vs-rest ROC,
paired detector-delta, and threshold-delta figures under `paper/figures/`.
The paired statistics are calculated over source images; restorations are not
treated as independent samples.

## Same-benchmark control run

To address part of the primary content confound, create a balanced GenImage
real/fake subset:

```powershell
python scripts/prepare_genimage_matched.py --pairs 500
python scripts/restore_images.py `
  --input data/matched_genimage/original `
  --output data/matched_genimage/restored `
  --device cuda
python scripts/make_manifest.py `
  --input data/matched_genimage `
  --output data/matched_genimage/manifest.csv
```

Train and evaluate the same binary probe on this manifest, and compare against
the COCO/GenImage result. Keep this control separate from the headline result:
the released data are balanced by benchmark but are not exact real/fake source
pairs.

For deployment-style transformations, generate JPEG and resize controls with
`scripts/make_postprocessing_variants.py`, then run the official evaluators
against each variant directory. These controls must be reported separately
from learned restoration because they are not restoration models.

Springer reviewers will expect the command, seed, Python/PyTorch versions, checkpoint hashes, image manifest, and hardware to be saved with the results.
