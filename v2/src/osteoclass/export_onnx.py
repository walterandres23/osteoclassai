"""Export a trained checkpoint to ONNX for in-browser inference (ONNX Runtime Web).

The exported graph returns ``logits`` (1, 5) and ``cam`` (1, 5, h, w): the classifier
head applied to every spatial location of the last feature map (class activation map).
Unlike Grad-CAM it needs no gradients, so it runs in the browser.

    python -m osteoclass.export_onnx --ckpt best.pt --out web/model.onnx --quantize \
        --check-root <dataset> --check-pred pred_test.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .gradcam import load_checkpoint


class CAMWrapper(nn.Module):
    """Logits + per-location class scores for timm models (ConvNeXt, ResNet...)."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        head = getattr(model, "head", None)
        self.norm = getattr(head, "norm", None) if head is not None else None
        self.fc = getattr(head, "fc", None) if head is not None else None
        if self.fc is None:                      # ResNet-style: model.fc
            self.fc = model.get_classifier()
        if self.norm is not None and not hasattr(self.norm, "normalized_shape"):
            self.norm = None

    def forward(self, x):
        feats = self.model.forward_features(x)
        logits = self.model.forward_head(feats)
        f = self.norm(feats) if self.norm is not None else feats       # LayerNorm2d keeps (B,C,H,W)
        # 1x1 convolution with the classifier weights = classifier applied at every location
        cam = nn.functional.conv2d(f, self.fc.weight[:, :, None, None], self.fc.bias)
        return logits, cam


def export(ckpt: str | Path, out: str | Path, opset: int = 17) -> Path:
    model, size = load_checkpoint(ckpt, "cpu")
    wrapper = CAMWrapper(model).eval()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(wrapper, dummy, out, input_names=["image"], output_names=["logits", "cam"],
                      opset_version=opset, dynamo=False)
    return out


def quantize(src: Path, dst: Path, min_size: int = 4096) -> Path:
    """Weight-only int8: store large weights as int8 + DequantizeLinear (per output channel).

    Compute stays in float32, so the graph only uses standard ops that ONNX Runtime Web
    supports (dynamic quantization emits ConvInteger, which the browser build lacks).
    The file is about 4x smaller.
    """
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(src))
    g = model.graph
    consumers: dict[str, list[tuple[str, int]]] = {}
    for n in g.node:
        for i, name in enumerate(n.input):
            consumers.setdefault(name, []).append((n.op_type, i))

    keep, new_nodes = [], []
    for init in g.initializer:
        w = numpy_helper.to_array(init)
        if w.dtype != np.float32 or w.ndim < 2 or w.size < min_size:
            keep.append(init)
            continue
        uses = consumers.get(init.name, [])
        axis = w.ndim - 1 if any(op == "MatMul" and i == 1 for op, i in uses) else 0
        red = tuple(d for d in range(w.ndim) if d != axis)
        scale = (np.abs(w).max(axis=red) / 127.0).astype(np.float32)
        scale[scale == 0] = 1.0
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        keep += [numpy_helper.from_array(q, init.name + "_q"),
                 numpy_helper.from_array(scale, init.name + "_scale"),
                 numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), init.name + "_zp")]
        new_nodes.append(helper.make_node("DequantizeLinear",
                                          [init.name + "_q", init.name + "_scale", init.name + "_zp"],
                                          [init.name], axis=axis, name=init.name + "_dq"))
    del g.initializer[:]
    g.initializer.extend(keep)
    nodes = new_nodes + list(g.node)
    del g.node[:]
    g.node.extend(nodes)
    onnx.checker.check_model(model)
    onnx.save(model, str(dst))
    return dst


def agreement(onnx_a: Path, onnx_b: Path, images: list[np.ndarray]) -> float:
    """Share of images where both models predict the same grade."""
    import onnxruntime as ort
    sa = ort.InferenceSession(str(onnx_a), providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(str(onnx_b), providers=["CPUExecutionProvider"])
    same = [sa.run(["logits"], {"image": x})[0].argmax() == sb.run(["logits"], {"image": x})[0].argmax()
            for x in images]
    return float(np.mean(same))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="web/model.onnx")
    ap.add_argument("--quantize", action="store_true", help="also write a weight-only int8 model (about 4x smaller, browser-compatible)")
    ap.add_argument("--check-root", help="dataset root, to compare fp32 vs int8 on test images")
    ap.add_argument("--check-pred", help="pred_test.csv with the image paths")
    ap.add_argument("--check-n", type=int, default=300)
    a = ap.parse_args(argv)

    fp32 = export(a.ckpt, a.out)
    print(f"fp32: {fp32} ({fp32.stat().st_size / 1e6:.1f} MB)")
    if not a.quantize:
        return
    int8 = quantize(fp32, fp32.with_name(fp32.stem + "_int8.onnx"))
    print(f"int8: {int8} ({int8.stat().st_size / 1e6:.1f} MB)")
    if a.check_root and a.check_pred:
        import pandas as pd
        from PIL import Image
        from .dataset import build_transforms
        tf = build_transforms(224)
        df = pd.read_csv(a.check_pred).sample(a.check_n, random_state=0)
        imgs = [tf(Image.open(Path(a.check_root) / p.replace("\\", "/")).convert("L")).unsqueeze(0).numpy()
                for p in df.path]
        print(f"fp32 vs int8 agreement on {len(imgs)} test images: {agreement(fp32, int8, imgs):.1%}")


if __name__ == "__main__":
    main()
