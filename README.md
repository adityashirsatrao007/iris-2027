# IRIS 27 Image Provenance Research Project

Reproducible code for **Original, Restored, or Generated: Image Restoration Pushes Authentic Photographs into the AI-Generated Class**.

This is an evaluation harness first, and a trainable three-class model second. It never writes invented accuracy values; results are computed only from supplied images and checkpoints.

## Simple explanation

This project asks a simple question:

> If a normal photograph is enhanced or restored, will an AI-image detector
> mistakenly think it was made by AI?

For each authentic photograph, the project creates a restored copy using an
image-restoration model. It then compares the original and restored versions
with AI-image detectors.

The project also includes a small three-label classifier that estimates whether
an image is:

- **Original** — an authentic image without restoration
- **Restored** — an authentic image processed by a restoration model
- **Generated** — an AI-generated image

The restoration does not change the scene or objects. It mainly changes fine
details such as noise, edges, and texture, so the two images can look almost
identical to a person.

The important finding is that restoration can change detector scores, but the
effect depends on the detector. Our official measurements did **not** reproduce
the very large false-positive rates claimed in the source abstract. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the measured results.

## Setup

```powershell
cd "C:\Users\aditya\Documents\Default Project"
& "C:\Users\aditya\documents-toolkit\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Put images under `data/raw/original`, `data/raw/restored`, and `data/raw/generated`. Then:

```powershell
python scripts/make_manifest.py --input data/raw --output data/manifest.csv --seed 42
python scripts/train_probe.py --manifest data/manifest.csv --output results/probe_128 --size 128
python scripts/make_report.py --predictions results/probe_128/predictions.csv --output results/probe_128
```

`scripts/run_all.ps1` reproduces the whole data build plus training from an empty checkout.

## Data actually used

| Class | Source | Count |
|---|---|---|
| original | COCO val2017 (`simopippa/reduced_coco_1000_val2017`) | 1,000 |
| restored | Real-ESRGAN `realesr-general-x4v3`, resampled to input size | 1,000 |
| generated | GenImage: BigGAN, VQDM, GLIDE, ADM (250 each) | 1,000 |

See `docs/DATASETS.md` for licenses, revisions, and limitations.

## Interactive demo

The recommended local presentation demo is a Gradio dashboard. It accepts one
image, creates a Real-ESRGAN preview, and displays the frozen three-class probe
alongside the released CNNSpot synthetic probability:

```powershell
& .\.venv-cuda\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\launch_demo.ps1 -Device cuda
```

Open `http://127.0.0.1:7860`. The first analyzed image downloads the official
Real-ESRGAN weight if it is not already present. The UI also shows an amplified
pixel-difference visualization and provides a **Clear local history** button.
History entries are labeled with the uploaded filename and paired `INPUT` and
`RESTORED` outputs. Re-uploading the same file is detected by SHA-256 and is
not saved as a duplicate.
Use `-Share` only when a temporary public link is needed. This UI is a
qualitative demonstration; the fixed paired results in `docs/RESULTS.md`
remain the authoritative measurements. Uploaded history is local and excluded
from Git.

## Models

Two training paths are provided:

- `scripts/train_probe.py` — frozen ImageNet ResNet-50 features + logistic-regression head. Runs in minutes on CPU and is the path used for the reported numbers.
- `scripts/train.py` — full fine-tuning of ResNet-50. Needs a GPU to be practical.

## Running on a GPU

The project supports CUDA when a compatible NVIDIA driver and CUDA-enabled
PyTorch are installed. The demo can use the local GPU with `--device cuda`.

Build the uploadable package once:

```powershell
Compress-Archive -Path scripts,src,notebooks,docs,paper,requirements.txt,README.md -DestinationPath dist/iris27_project.zip -Force
```

Then upload `dist/iris27_project.zip` to `notebooks/iris27_colab.ipynb` in Colab
(Runtime → GPU). The notebook installs dependencies, downloads the corpora,
restores, trains, and hands back a zip of `results/`. On a T4 the whole run is
about 25 minutes.

On any CUDA host the driver can also be called directly:

```bash
python notebooks/colab_run.py --stage all --epochs 8
```

Stages are idempotent, so `--stage restore` or `--stage finetune` reruns only that
part. `scripts/restore_images.py` and `scripts/restore_restormer.py` accept
`--device {auto,cpu,cuda}`; `scripts/train.py` and `scripts/train_probe.py` select
CUDA automatically.

## Importing a Hugging Face image dataset

Install the extra requirements above, then inspect a public dataset and import its image column:

```powershell
python scripts/import_hf.py --dataset OWNER/DATASET --split train --image-column image --output data/raw/generated
```

Use this only when the dataset documentation confirms that the images are AI-generated. The importer records the dataset ID, revision, split, and row index in `metadata.jsonl`; retain that file for the paper's data statement. Kaggle can also be used, but it requires a Kaggle API token and its license/terms must be checked before redistribution. Do not mix datasets or labels without documenting the source.

See `docs/EXPERIMENT_PROTOCOL.md` before collecting paper results. The exact CNNSpot, UnivFD, Restormer, and Real-ESRGAN checkpoints must be obtained from their official repositories and used according to their licenses.

## Results

Pilot results on 3,000 images (1,000 per class) are in [`docs/RESULTS.md`](docs/RESULTS.md).
Headline numbers for the reproducible 224-pixel probe:

| Input | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|
| 128 px | 82.29% | 81.62% | 0.926 |
| 224 px | 88.28% | 87.77% | 0.957 |

Unseen-generator mean AUROC: 0.978. The central abstract claim (restoration
inflating the false-positive rate) is **not** reproduced; see the official
CNNSpot and UnivFD audit in [`docs/RESULTS.md`](docs/RESULTS.md).

## Scientific safeguards

- Split by source image, never by patches or augmented copies.
- Hold out at least one restoration operator and generator family.
- Report macro-F1, balanced accuracy, per-class recall, AUROC, calibration, and FPR at fixed operating points; accuracy alone is insufficient.
- Save model versions, preprocessing, seeds, hardware, and the complete manifest.
