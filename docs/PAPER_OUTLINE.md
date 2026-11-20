# Springer LNEE paper outline

Target: IRIS'27 proceedings, Springer *Lecture Notes in Electrical Engineering*.
Minimum length is four pages; abstracts alone are not published.

## Suggested structure

1. **Introduction** — deployed detectors assume a binary world; phone cameras
   and editors now run learned restoration, creating a third provenance state.
2. **Related work** — CNNSpot, UnivFD, GenImage, Real-ESRGAN, Restormer.
3. **Method**
   - 3.1 Provenance taxonomy: original / restored / generated.
   - 3.2 Detector scoring protocol and the published 0.5 threshold.
   - 3.3 Learned restoration operators and resampling back to input size.
   - 3.4 Frozen-backbone linear probe for the three-class task.
4. **Experimental setup** — datasets, split rule, hardware, seeds, hyperparameters.
5. **Results**
   - 5.1 False-positive rate on authentic images before and after restoration.
   - 5.2 Three-class provenance classification (accuracy, macro-F1, per class).
   - 5.3 Binary contrasts: original vs generated, original vs restored.
   - 5.4 Calibration and threshold transfer.
   - 5.5 Cross-generator holdout.
6. **Discussion** — restoration is a provenance class, not a nuisance transform.
7. **Threats to validity** — content confound between COCO and GenImage, single
   restorer, 128 px ceiling, CPU-only compute.
8. **Conclusion and recommendations** — benchmark restoration explicitly;
   recalibrate thresholds; report FPR at the deployed operating point.
9. **Data and code availability** — manifest, scripts, seeds, checkpoint hashes.

## Required tables and figures

| Item | Source |
|---|---|
| Dataset composition | `docs/DATASETS.md` |
| Main metrics | `results/*/metrics.json` |
| Per-class report | `results/*/REPORT.md` |
| Confusion matrix | `results/*/REPORT.md` |
| FPR before/after restoration | `results/*/REPORT.md` |
| Reproducibility statement | `scripts/run_all.ps1` |

## Claims to avoid

- Do not call the result state of the art; this is an evaluation study.
- Do not generalise to Restormer or to diffusion generators that were not run.
- Do not report accuracy without the class balance and the split rule.
- Do not compare against CNNSpot/UnivFD scores unless those checkpoints were
  actually executed in this environment.
