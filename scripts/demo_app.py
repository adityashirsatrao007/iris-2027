"""Interactive IRIS 27 image-provenance demo.

This is a presentation tool, not a replacement for the paired evaluation
protocol. The three-class probe is the frozen ImageNet ResNet-50 probe trained
by ``scripts/train_probe.py``; CNNSpot is the released, untouched checkpoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "CNNDetection"))
from networks.resnet import resnet50  # noqa: E402


LABELS = ["Original", "Restored", "Generated"]


def load_probe(path: Path, device: torch.device):
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone.fc = torch.nn.Identity()
    backbone.eval().to(device)
    classifier = joblib.load(path)
    return backbone, classifier


def load_cnnspot(path: Path, device: torch.device):
    model = resnet50(num_classes=1)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    return model.eval().to(device)


def build_app(probe_path: Path, cnnspot_path: Path, device_name: str):
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else ("cpu" if device_name == "auto" else device_name)
    )
    if not probe_path.exists():
        raise FileNotFoundError(
            f"{probe_path} is missing. Run scripts/train_probe.py first."
        )
    if not cnnspot_path.exists():
        raise FileNotFoundError(f"Missing official CNNSpot checkpoint: {cnnspot_path}")

    backbone, classifier = load_probe(probe_path, device)
    cnnspot = load_cnnspot(cnnspot_path, device)
    probe_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    detector_tf = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    restorer = None
    weights = ROOT / "data" / "weights" / "realesr-general-x4v3.pth"

    def restore(image: Image.Image) -> Image.Image:
        nonlocal restorer
        if restorer is None:
            weights.parent.mkdir(parents=True, exist_ok=True)
            if not weights.exists():
                import urllib.request
                urllib.request.urlretrieve(
                    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
                    "v0.2.5.0/realesr-general-x4v3.pth", weights
                )
            restorer = ModelLoader().load_from_file(str(weights)).model.eval().to(device)
        source = image.convert("RGB")
        arr = np.asarray(source).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.inference_mode():
            output = restorer(tensor).clamp(0, 1)[0].permute(1, 2, 0)
        result = Image.fromarray((output.cpu().numpy() * 255).round().astype(np.uint8))
        return result.resize(source.size, Image.Resampling.LANCZOS)

    def score(image: Image.Image):
        if image is None:
            return None, None, "Upload an image to begin."
        image = image.convert("RGB")
        restored = restore(image)
        with torch.inference_mode():
            features = backbone(probe_tf(image).unsqueeze(0).to(device)).cpu().numpy()
            probe_probs = classifier.predict_proba(features)[0]
            cnn_score = float(cnnspot(detector_tf(image).unsqueeze(0).to(device))
                              .sigmoid().item())
            restored_score = float(cnnspot(
                detector_tf(restored).unsqueeze(0).to(device)
            ).sigmoid().item())
        probe_text = "\n".join(
            f"- **{label}:** {prob:.1%}" for label, prob in zip(LABELS, probe_probs)
        )
        detector_text = (
            f"- **CNNSpot synthetic probability (input):** {cnn_score:.1%}\n"
            f"- **CNNSpot synthetic probability (Real-ESRGAN preview):** "
            f"{restored_score:.1%}\n"
            f"- **Decision threshold:** 50%\n"
        )
        note = (
            f"### IRIS 27 result\n{probe_text}\n\n### Official CNNSpot score\n"
            f"{detector_text}\n"
            f"_Device: `{device}`. Scores are model outputs, not proof of provenance._"
        )
        return image, restored, note

    with gr.Blocks(title="IRIS 27 — Image Provenance") as demo:
        gr.Markdown(
            "# IRIS 27: Original, Restored, or Generated?\n"
            "Upload one image to inspect the frozen three-class probe and the "
            "released CNNSpot detector."
        )
        with gr.Row():
            with gr.Column():
                upload = gr.Image(type="pil", label="Input image")
                run = gr.Button("Analyze image", variant="primary")
            with gr.Column():
                original = gr.Image(label="Input")
                restored_output = gr.Image(label="Real-ESRGAN preview")
        report = gr.Markdown("Upload an image, then click **Analyze image**.")
        run.click(score, inputs=upload, outputs=[original, restored_output, report])
        gr.Markdown(
            "**Research caution:** this interactive view is a qualitative demo. "
            "The paper's measured rates come from the fixed 1,000-image paired "
            "protocol in `docs/RESULTS.md`, not from this single-image UI."
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", default="results/demo_probe_224/classifier.joblib")
    parser.add_argument(
        "--cnnspot",
        default="data/official_models/cnnspot_blur_jpg_prob0.5.pth",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app(ROOT / args.probe, ROOT / args.cnnspot, args.device)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
