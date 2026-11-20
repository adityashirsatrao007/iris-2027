# Research-gap and novelty audit

**Checked:** 20 November 2026  
**Scope:** AI-generated image detection, restoration/enhancement, and
real-world robustness.

## Short answer

The project should **not** claim that it is the first work to use restoration
or other transformations for AI-image detection. Recent work already studies
transform-induced residuals as detection evidence.

The defensible contribution is narrower:

> It treats a learned restoration of an authentic photograph as a distinct
> provenance state and measures, with source-paired images, whether released
> binary detectors falsely call that state synthetic.

That is different from training a detector to exploit restoration residuals in
order to classify fake versus real images.

## What prior work already covers

### GenImage (NeurIPS 2023)

GenImage introduced a large real/fake benchmark and evaluates cross-generator
generalization and degraded-image classification (low resolution, blur, and
JPEG). It establishes that post-processing and generator changes matter, but
its primary label remains real versus generated; it does not make
“authentic-but-restored” a third provenance label.

Source: <https://arxiv.org/abs/2306.08571>

### RRDataset / RRBench (ICCV 2025)

This work evaluates 17 detectors and 10 VLMs under realistic transmission and
re-digitization pipelines. It is highly relevant to deployment robustness.
However, its focus is real-world image transformation, not the specific
question of a learned restoration operator applied to the same authentic
source and its effect on detector false positives.

Source: <https://arxiv.org/abs/2509.09172>

### GenRes / GenRes++ (2026 preprint)

GenRes and GenRes++ apply multiple generative transformations, including
restoration, super-resolution, enhancement, and denoising, then learn
relational residuals for binary real-versus-AI detection. This directly overlaps
the broad idea that transformations reveal generative traces.

The distinction from IRIS 27 is the target question: GenRes seeks better
fake-image detection, whereas IRIS 27 measures whether **authentic restored
images are wrongly flagged** and explicitly models original/restored/generated
as separate states.

Source: <https://arxiv.org/abs/2607.08674>

### Detector shortcut and robustness work

Recent work also shows that detectors rely on global or low-level artifacts and
can fail after carefully chosen image transformations. This supports the
motivation for an audit, but it makes strong novelty claims about detector
fragility unsafe.

Examples:

- <https://arxiv.org/abs/2602.00192>
- <https://arxiv.org/abs/2605.14486>
- <https://arxiv.org/abs/2605.16122>

## What is genuinely distinctive in this project

1. **A provenance taxonomy rather than only a detector label.** The labels are
   original, restored-authentic, and generated.
2. **Source-paired evaluation.** Each restored image is tied to the exact
   authentic source, allowing before/after score changes instead of comparing
   unrelated image populations.
3. **Operational false-positive question.** The project reports the fraction of
   authentic images crossing a fixed detector threshold after restoration.
4. **Released-checkpoint audit.** CNNSpot and UnivFD are evaluated without
   retraining, and the abstract’s numerical claims are tested rather than
   repeated.
5. **A negative and detector-dependent result.** The measured outcome is not
   “restoration always fools detectors”; it varies by detector checkpoint and
   restoration operator.

The strongest paper framing is therefore **an evaluation protocol and
reproducibility audit for restoration-induced provenance errors**, not a new
detector architecture.

## Remaining research gaps

These are real open gaps, not gaps that can be solved by rewriting the claim:

1. **Matched real/fake content.** COCO authentic images and GenImage images have
   different content distributions. A matched real split is needed to separate
   synthesis evidence from content shortcuts.
2. **More restoration operators.** Two operators are informative but cannot
   establish generality. Add at least one denoiser, deblurrer, and camera/ISP
   pipeline, with one operator held out.
3. **Detector operating curves.** The current audit uses score 0.5. A complete
   study should report ROC/precision-recall curves, calibration, AUROC, and
   FPR at several fixed operating points.
4. **Statistical stability.** The probe needs multiple seeds and bootstrap
   confidence intervals for paired detector deltas.
5. **Modern detector coverage.** CNNSpot and UnivFD are important released
   baselines, but newer detectors and commercial/VLM systems should be tested
   separately.
6. **Real deployment transformations.** Social-media recompression,
   screenshotting, re-digitization, and camera ISP processing should be
   compared with learned restoration.
7. **Causal attribution.** A difference map or frequency analysis can show
   where restoration changes pixels, but it does not prove which artifact a
   detector uses. Controlled ablations are needed.

## Recommended claim

Use this wording:

> We introduce a source-paired evaluation of learned restoration as an
> overlooked provenance state. On a controlled corpus, we compare original
> authentic photographs with restored copies using a three-class probe and
> released binary detectors. The results show detector-dependent score shifts
> and do not reproduce the motivating abstract’s large false-positive rates.

Avoid:

- “first restoration-based AI detector”
- “restoration universally fools detectors”
- “88% accuracy proves provenance”
- “the abstract is false in general”

The results support a **measured audit on this corpus**, not a universal claim
about every restorer, detector, dataset, or deployment setting.
