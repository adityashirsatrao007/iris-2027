# IRIS 27 Image Provenance Research Project

Reproducible code for **Original, Restored, or Generated: Image Restoration Pushes Authentic Photographs into the AI-Generated Class**.

This is an evaluation harness first, and a trainable three-class model second. It never writes invented accuracy values; results are computed only from supplied images and checkpoints.

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

## Models

Two training paths are provided:

- `scripts/train_probe.py` — frozen ImageNet ResNet-50 features + logistic-regression head. Runs in minutes on CPU and is the path used for the reported numbers.
- `scripts/train.py` — full fine-tuning of ResNet-50. Needs a GPU to be practical.

## Running on a GPU

This machine has no CUDA device, so all reported numbers come from CPU runs. The
pipeline is GPU-ready and the same code runs unchanged on a GPU host.

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
Headline numbers, all computed here on CPU:

| Input | Accuracy | Macro-F1 | AUROC |
|---|---|---|---|
| 128 px | 82.29% | 81.62% | 0.926 |
| 224 px | 88.28% | 87.77% | 0.957 |

Unseen-generator mean AUROC: 0.978. The central abstract claim (restoration
inflating the false-positive rate) is **not** reproduced by this generic
classifier and still requires the official CNNSpot and UnivFD checkpoints.

## Scientific safeguards

- Split by source image, never by patches or augmented copies.
- Hold out at least one restoration operator and generator family.
- Report macro-F1, balanced accuracy, per-class recall, AUROC, calibration, and FPR at fixed operating points; accuracy alone is insufficient.
- Save model versions, preprocessing, seeds, hardware, and the complete manifest.
