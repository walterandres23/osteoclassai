"""OsteoClassAI demo: knee X-ray -> Kellgren-Lawrence grade + Grad-CAM.

Runs on a free CPU Hugging Face Space. Expects the checkpoint written by
``osteoclass.train`` at ``MODEL_PATH`` (default: best.pt next to this file).
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
import torch
from PIL import Image

from osteoclass.dataset import build_transforms
from osteoclass.gradcam import GradCAM, load_checkpoint

torch.set_num_threads(max(1, os.cpu_count() or 1))
HERE = Path(__file__).parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", HERE / "best.pt"))

GRADES = {
    0: "KL0 · No osteoarthritis",
    1: "KL1 · Doubtful (possible osteophyte)",
    2: "KL2 · Minimal (definite osteophyte, possible narrowing)",
    3: "KL3 · Moderate (multiple osteophytes, definite narrowing)",
    4: "KL4 · Severe (large osteophytes, marked narrowing, sclerosis)",
}

model, SIZE = load_checkpoint(MODEL_PATH, "cpu")
cam = GradCAM(model)
tf = build_transforms(SIZE)
JET = matplotlib.colormaps["jet"]


def overlay(gray: np.ndarray, heat: np.ndarray, alpha: float = 0.4) -> Image.Image:
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32) / 255
    color = JET(heat)[..., :3]
    return Image.fromarray((255 * ((1 - alpha) * base + alpha * color)).clip(0, 255).astype(np.uint8))


def predict(img: Image.Image):
    if img is None:
        return None, None, "Upload a knee X-ray."
    gray = img.convert("L")
    heat, probs = cam(tf(gray).unsqueeze(0))
    k = int(probs.argmax())
    oa = float(probs[2:].sum())
    shown = np.asarray(gray.resize((SIZE, SIZE)))
    label = {GRADES[i]: float(p) for i, p in enumerate(probs)}
    text = (f"### Predicted: {GRADES[k]}\n"
            f"Confidence {probs[k]:.0%} · Probability of osteoarthritis (KL ≥ 2): **{oa:.0%}**\n\n"
            "The heat map shows the regions that drove the prediction (Grad-CAM).")
    return label, overlay(shown, heat), text


EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*.png"))

with gr.Blocks(title="OsteoClassAI") as demo:
    gr.Markdown(
        "# OsteoClassAI · Knee osteoarthritis grading\n"
        "Upload a **posteroanterior knee X-ray cropped to the joint** (like the OAI images). "
        "ConvNeXt-Tiny trained on 5,778 knees; on a patient-disjoint test set: QWK 0.851, "
        "AUC 0.943 for KL ≥ 2. [Code and evaluation](https://github.com/walterandres23/osteoclassai)"
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="pil", label="Knee X-ray", image_mode="L", height=320)
            btn = gr.Button("Grade", variant="primary")
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp, label="Examples (OAI test set)")
        with gr.Column():
            txt = gr.Markdown()
            out_label = gr.Label(num_top_classes=5, label="KL grade probabilities")
            out_img = gr.Image(label="Grad-CAM", height=320)
    btn.click(predict, inp, [out_label, out_img, txt])
    inp.upload(predict, inp, [out_label, out_img, txt])
    gr.Markdown(
        "⚠️ **Research and educational demo. Not a medical device**; it must not be used "
        "for diagnosis. Images are processed in memory and not stored."
    )

if __name__ == "__main__":
    demo.launch()
