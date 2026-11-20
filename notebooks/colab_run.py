"""End-to-end IRIS 27 pipeline for a GPU host (Colab, Kaggle, or a local CUDA box).

Every number in the paper is produced by this script; it only orchestrates the
repo's own scripts, so a GPU run and the CPU pilot differ in speed, not in method.

Stages can be run individually:

    python colab_run.py --stage data        # download COCO + GenImage
    python colab_run.py --stage restore     # Real-ESRGAN + Restormer
    python colab_run.py --stage probe       # frozen-backbone probe at 224 and 128
    python colab_run.py --stage finetune    # full ResNet-50 fine-tune
    python colab_run.py --stage protocols   # cross-generator + unseen restorer
    python colab_run.py --stage all         # everything, in order

The script is idempotent: a stage whose outputs already exist is skipped unless
--force is passed.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DL = ROOT / "data" / "downloads"
RES = ROOT / "results"

COCO = "simopippa/reduced_coco_1000_val2017"
COCO_SPLITS = (("train", 800, "cocoval"), ("validation", 100, "cocovalb"), ("test", 100, "cocovalc"))
GENERATORS = ("BigGAN", "VQDM", "glide", "ADM")


def run(cmd, **kwargs):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, **kwargs)


def py(*args):
    return [sys.executable, *[str(a) for a in args]]


def count(pattern):
    return len(list(RAW.glob(pattern)))


# ---------------------------------------------------------------- data
def stage_data(force=False):
    (DL / "coco").mkdir(parents=True, exist_ok=True)
    (RAW / "generated").mkdir(parents=True, exist_ok=True)

    if count("original/*.jpg") >= 1000 and not force:
        print(f"originals: {count('original/*.jpg')} already present")
    else:
        for split, limit, prefix in COCO_SPLITS:
            parquet = DL / "coco" / f"{split}.parquet"
            if not parquet.exists():
                run(["curl", "-sL", "--retry", "10", "--retry-all-errors", "-o", str(parquet),
                     f"https://huggingface.co/datasets/{COCO}/resolve/main/data/{split}-00000-of-00001.parquet"])
            run(py("scripts/prepare_coco_parquet.py", "--parquet", parquet,
                   "--output", RAW / "original", "--prefix", prefix,
                   "--size", 256, "--limit", limit))
        print(f"originals: {count('original/*.jpg')}")

    for gen in GENERATORS:
        out = RAW / "generated" / f"genimage_{gen.lower()}"
        have = len(list(out.glob("*.jpg"))) + len(list(out.glob("*.png")))
        if have >= 250 and not force:
            print(f"generated {gen}: {have} already present")
            continue
        run(py("scripts/import_hf.py", "--dataset", f"bitmind/GenImage_{gen}", "--split", "train",
               "--output", out, "--limit", 250, "--prefix", gen.lower()))


# ------------------------------------------------------------- restore
def stage_restore(force=False):
    if count("restored/*.jpg") >= 1000 and not force:
        print(f"restored (Real-ESRGAN): {count('restored/*.jpg')} already present")
    else:
        run(py("scripts/restore_images.py", "--input", RAW / "original",
               "--output", RAW / "restored", "--model", "general-x4v3", "--device", "auto"))

    if count("restored_restormer/*.jpg") >= 1000 and not force:
        print(f"restored (Restormer): {count('restored_restormer/*.jpg')} already present")
    else:
        run(py("scripts/restore_restormer.py", "--input", RAW / "original",
               "--output", RAW / "restored_restormer", "--model", "real-denoising",
               "--device", "auto"))


# ------------------------------------------------------------ manifest
def stage_manifest(force=False):
    path = ROOT / "data" / "manifest.csv"
    if path.exists() and not force:
        print(f"manifest exists: {path}")
        return
    run(py("scripts/make_manifest.py", "--input", RAW, "--output", path, "--seed", "42"))


# --------------------------------------------------------------- probe
def stage_probe(force=False):
    RES.mkdir(parents=True, exist_ok=True)
    for size in (224, 128):
        out = RES / f"probe_{size}"
        if (out / "metrics.json").exists() and not force:
            print(f"probe {size}: already present")
            continue
        run(py("scripts/train_probe.py", "--manifest", ROOT / "data" / "manifest.csv",
               "--output", out, "--size", size, "--batch-size", 64))
        run(py("scripts/make_report.py", "--predictions", out / "predictions.csv", "--output", out))


# ------------------------------------------------------------ finetune
def stage_finetune(epochs=8, size=224, force=False):
    out = RES / f"finetune_{size}"
    if (out / "metrics.json").exists() and not force:
        print(f"finetune {size}: already present")
        return
    run(py("scripts/train.py", "--manifest", ROOT / "data" / "manifest.csv",
           "--output", out, "--epochs", epochs, "--size", size,
           "--batch-size", 64, "--workers", 2))
    run(py("scripts/make_report.py", "--predictions", out / "predictions.csv", "--output", out))


# ----------------------------------------------------------- protocols
def stage_protocols(size=128, force=False):
    tag = f"features_all_{size}.npz"
    if not (RES / tag).exists() or force:
        run(py("scripts/extract_features.py", "--manifest", ROOT / "data" / "manifest.csv",
               "--output", RES / tag, "--size", size, "--batch-size", 64))
    run(py("scripts/evaluate_protocols.py", "--features", RES / tag,
           "--output", RES / f"protocols_{size}"))


# ------------------------------------------------------------- figures
def stage_figures(predictions="probe_224", force=False):
    run(py("scripts/make_figures.py",
           "--predictions", RES / predictions / "predictions.csv",
           "--protocols", RES / "protocols_128" / "protocols.json",
           "--output", RES / "figures"))


# ------------------------------------------------------------- reports
def stage_env(force=False):
    if (RES / "ENVIRONMENT.json").exists() and not force:
        print("environment record exists")
    else:
        run(py("scripts/env_report.py"))


STAGES = {
    "env": stage_env,
    "data": stage_data,
    "restore": stage_restore,
    "manifest": stage_manifest,
    "probe": stage_probe,
    "finetune": stage_finetune,
    "protocols": stage_protocols,
    "figures": stage_figures,
}
ORDER = ["env", "data", "restore", "manifest", "probe", "finetune", "protocols", "figures"]

p = argparse.ArgumentParser()
p.add_argument("--stage", default="all", choices=["all", *ORDER])
p.add_argument("--force", action="store_true")
p.add_argument("--epochs", type=int, default=8, help="fine-tune epochs")
a = p.parse_args()

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "180")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "180")

for name in (ORDER if a.stage == "all" else [a.stage]):
    print(f"\n{'=' * 70}\n== {name}\n{'=' * 70}", flush=True)
    kwargs = {"force": a.force}
    if name == "finetune":
        kwargs["epochs"] = a.epochs
    STAGES[name](**kwargs)

print("\nPipeline complete. Artifacts in results/")
