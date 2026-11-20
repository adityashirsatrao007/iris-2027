# Full reproduction pipeline. Run from the project root.
$ErrorActionPreference = "Stop"
$py = "C:\Users\aditya\documents-toolkit\.venv\Scripts\python.exe"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "180"
$env:HF_HUB_ETAG_TIMEOUT = "180"

# 1,000 authentic COCO val2017 images (mirror of the official val2017 subset)
& $py scripts\import_hf.py --dataset simopippa/reduced_coco_1000_val2017 --split train `
    --output data\raw\original --limit 1000 --prefix cocoval

# 1,000 generated images, 250 from each GenImage generator family
foreach ($g in @("BigGAN", "VQDM", "glide", "ADM")) {
    & $py scripts\import_hf.py --dataset "bitmind/GenImage_$g" --split train `
        --output "data\raw\generated\genimage_$g" --limit 250 --prefix $g.ToLower()
}

# Learned restoration, resampled back to the input dimensions
& $py scripts\restore_images.py --input data\raw\original --output data\raw\restored --model general-x4v3

& $py scripts\make_manifest.py --input data\raw --output data\manifest.csv --seed 42

# Primary result: resolution matched to the 128 px GenImage subset
& $py scripts\train_probe.py --manifest data\manifest.csv --output results\probe_128 --size 128
& $py scripts\make_report.py --predictions results\probe_128\predictions.csv --output results\probe_128

# Secondary result at the standard detector input size
& $py scripts\train_probe.py --manifest data\manifest.csv --output results\probe_224 --size 224
& $py scripts\make_report.py --predictions results\probe_224\predictions.csv --output results\probe_224
