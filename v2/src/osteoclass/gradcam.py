"""Grad-CAM heat maps for a trained checkpoint.

    python -m osteoclass.gradcam --ckpt outputs/runs/convnext_tiny/best.pt \
        --root <dataset> --pred outputs/runs/convnext_tiny/pred_test.csv --out media/gradcam_test.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from .dataset import build_transforms


def default_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Last spatial feature block: ConvNeXt/EfficientNet stages, ResNet layer4, else last Conv2d."""
    if hasattr(model, "stages"):
        return model.stages[-1]
    if hasattr(model, "layer4"):
        return model.layer4
    convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    return convs[-1]


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module | None = None):
        self.model = model.eval()
        layer = layer or default_target_layer(model)
        self._act = self._grad = None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "_act", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grad", go[0]))

    def __call__(self, x: torch.Tensor, cls: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return (heat map in [0, 1] at input size, class probabilities)."""
        x = x.requires_grad_(True)
        logits = self.model(x)
        probs = torch.softmax(logits, 1)[0].detach().cpu().numpy()
        cls = int(probs.argmax()) if cls is None else cls
        self.model.zero_grad(set_to_none=True)
        logits[0, cls].backward()
        w = self._grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self._act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = (cam / (cam.max() + 1e-8)).detach().cpu().numpy()
        return cam, probs


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, int]:
    ck = torch.load(path, map_location=device)
    model = timm.create_model(ck["model"], pretrained=False, num_classes=5)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck.get("size", 224)


def pick_examples(pred: pd.DataFrame, per_grade: int = 1, seed: int = 0) -> pd.DataFrame:
    """One correctly classified test knee per KL grade (random, reproducible)."""
    p = pred[[f"p{k}" for k in range(5)]].to_numpy()
    ok = pred[p.argmax(1) == pred.kl.to_numpy()]
    parts = [g.sample(min(per_grade, len(g)), random_state=seed) for _, g in ok.groupby("kl")]
    return pd.concat(parts).sort_values("kl").reset_index(drop=True)


def grid(model, size, root, rows: pd.DataFrame, out: Path, device="cpu") -> None:
    cam = GradCAM(model)
    tf = build_transforms(size)
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6))
    axes = np.atleast_2d(axes).reshape(2, n)
    for j, r in rows.iterrows():
        img = Image.open(Path(root) / str(r.path).replace("\\", "/")).convert("L")
        heat, probs = cam(tf(img).unsqueeze(0).to(device))
        show = np.asarray(img.resize((size, size)))
        axes[0, j].imshow(show, cmap="gray")
        axes[0, j].set_title(f"KL{int(r.kl)}", fontsize=12)
        axes[1, j].imshow(show, cmap="gray")
        axes[1, j].imshow(heat, cmap="jet", alpha=0.4)
        axes[1, j].set_title(f"pred KL{probs.argmax()} ({probs.max():.0%})", fontsize=10)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle("Grad-CAM, correctly classified test knees", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--pred", required=True, help="pred_test.csv from train.py")
    ap.add_argument("--out", default="media/gradcam_test.png")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_checkpoint(a.ckpt, device)
    rows = pick_examples(pd.read_csv(a.pred), seed=a.seed)
    grid(model, size, a.root, rows, Path(a.out), device)
    print("saved", a.out)


if __name__ == "__main__":
    main()
