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

Springer reviewers will expect the command, seed, Python/PyTorch versions, checkpoint hashes, image manifest, and hardware to be saved with the results.
