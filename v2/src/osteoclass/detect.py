"""Knee-joint detector (CenterNet-style) trained on Chen's DetKneeData, and the crop rule of the classifiers.

The detector sees the radiograph letterboxed to 256 × 320 (gray, repeated to 3 channels) and predicts, at stride 4:
a heatmap of knee centers (1 channel), the sub-pixel offset of each center (2) and log(box width / 4) (1).
A knee crop for the classifiers is then the square of side ``CROP_FACTOR`` × box width with the same center, which
is how the KneeKL224 crops relate to Chen's boxes (measured in ``osteoclass.detdata``: median 1.30, center offset
under 1% of the box). Mirroring is not used by the crops; the patient's right knee is on the left of the image.

Training augmentation imitates other sources: single-knee views with the joint larger in the frame (as in Gornale
and in phone photos of a film), scale and position changes, mirroring, contrast, gamma, blur and noise.

Usage:
    python -m osteoclass.detect --data outputs/det --out runs/detector [--epochs 40]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IN_H, IN_W, STRIDE = 256, 320, 4
OUT_H, OUT_W = IN_H // STRIDE, IN_W // STRIDE
CROP_FACTOR = 1.30
MEAN, STD = 0.5, 0.25


# ---------- geometry (numpy, no torch) ----------

def letterbox(gray: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Resize keeping the aspect ratio to fit IN_H × IN_W and pad with black; returns image, scale, pad x, pad y."""
    h, w = gray.shape
    s = min(IN_H / h, IN_W / w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = np.zeros((IN_H, IN_W), np.uint8)
    px, py = (IN_W - nw) // 2, (IN_H - nh) // 2
    out[py:py + nh, px:px + nw] = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    return out, s, px, py


def make_targets(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Heatmap (OUT_H, OUT_W), offsets (2, OUT_H, OUT_W), log-width (OUT_H, OUT_W) and mask of center cells."""
    heat = np.zeros((OUT_H, OUT_W), np.float32)
    off = np.zeros((2, OUT_H, OUT_W), np.float32)
    logw = np.zeros((OUT_H, OUT_W), np.float32)
    mask = np.zeros((OUT_H, OUT_W), np.float32)
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / STRIDE, (y1 + y2) / 2 / STRIDE
        w = (x2 - x1) / STRIDE
        ix, iy = int(cx), int(cy)
        if not (0 <= ix < OUT_W and 0 <= iy < OUT_H) or w <= 0:
            continue
        sigma = max(0.8, w / 8)
        heat = np.maximum(heat, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
        heat[iy, ix] = 1.0
        off[:, iy, ix] = (cx - ix, cy - iy)
        logw[iy, ix] = math.log(w)
        mask[iy, ix] = 1.0
    return heat, off, logw, mask


def decode(heat: np.ndarray, off: np.ndarray, logw: np.ndarray, k: int = 2, thr: float = 0.3) -> list[dict]:
    """Peaks of the (already sigmoid) heatmap → knees as center x, y and box width, in letterboxed pixels."""
    pad = np.pad(heat, 1, constant_values=-1)
    neigh = np.max(np.stack([pad[dy:dy + OUT_H, dx:dx + OUT_W] for dy in range(3) for dx in range(3)]), 0)
    peaks = np.argwhere((heat >= neigh) & (heat >= thr))
    order = np.argsort(-heat[peaks[:, 0], peaks[:, 1]])[:k]
    knees = []
    for iy, ix in peaks[order]:
        cx = (ix + off[0, iy, ix]) * STRIDE
        cy = (iy + off[1, iy, ix]) * STRIDE
        knees.append({"score": float(heat[iy, ix]), "cx": float(cx), "cy": float(cy),
                      "w": float(math.exp(logw[iy, ix]) * STRIDE)})
    return sorted(knees, key=lambda d: d["cx"])


def to_original(knee: dict, scale: float, px: float, py: float) -> dict:
    return {**knee, "cx": (knee["cx"] - px) / scale, "cy": (knee["cy"] - py) / scale, "w": knee["w"] / scale}


def crop_box(cx: float, cy: float, w: float, factor: float = CROP_FACTOR) -> tuple[float, float, float, float]:
    s = factor * w
    return cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def augment(gray: np.ndarray, boxes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random view of a training radiograph (gray, boxes x1 y1 x2 y2) → letterboxed image and boxes."""
    h, w = gray.shape
    if rng.random() < 0.4:
        # single-knee view: the joint fills 25-55% of the width, at a random position
        b = boxes[rng.integers(len(boxes))]
        bw = b[2] - b[0]
        cw = bw / rng.uniform(0.25, 0.55)
        ch = cw * rng.uniform(0.8, 1.5)
        cx = (b[0] + b[2]) / 2 + rng.uniform(-0.2, 0.2) * cw
        cy = (b[1] + b[3]) / 2 + rng.uniform(-0.2, 0.2) * ch
    else:
        # whole radiograph, zoomed in or out and shifted
        z = rng.uniform(0.8, 1.35)
        cw, ch = w * z, h * z
        cx, cy = w / 2 + rng.uniform(-0.1, 0.1) * w, h / 2 + rng.uniform(-0.1, 0.1) * h
    x0, y0 = cx - cw / 2, cy - ch / 2
    sx, sy = IN_W / cw, IN_H / ch
    s = min(sx, sy)  # keep the aspect: letterbox inside the window
    ox, oy = (IN_W - cw * s) / 2, (IN_H - ch * s) / 2
    m = np.array([[s, 0, ox - x0 * s], [0, s, oy - y0 * s]], np.float32)
    img = cv2.warpAffine(gray, m, (IN_W, IN_H), flags=cv2.INTER_LINEAR, borderValue=0)
    nb = boxes.copy().astype(np.float32)
    nb[:, [0, 2]] = nb[:, [0, 2]] * s + m[0, 2]
    nb[:, [1, 3]] = nb[:, [1, 3]] * s + m[1, 2]
    # keep a knee only if its center is inside and at least 60% of the box is visible
    keep = []
    for bb in nb:
        cxx, cyy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        vis = (max(0, min(bb[2], IN_W) - max(bb[0], 0)) * max(0, min(bb[3], IN_H) - max(bb[1], 0))
               / max(1e-6, (bb[2] - bb[0]) * (bb[3] - bb[1])))
        keep.append(0 <= cxx < IN_W and 0 <= cyy < IN_H and vis >= 0.6)
    nb = nb[np.array(keep, bool)] if len(nb) else nb
    if rng.random() < 0.5:
        img = img[:, ::-1].copy()
        nb[:, [0, 2]] = IN_W - nb[:, [2, 0]]
    # photometric: contrast, brightness, gamma, blur, noise
    f = img.astype(np.float32) / 255
    f = np.clip((f - 0.5) * rng.uniform(0.7, 1.4) + 0.5 + rng.uniform(-0.15, 0.15), 0, 1)
    f = f ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.3:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.5))
    if rng.random() < 0.3:
        f = np.clip(f + rng.normal(0, rng.uniform(0.01, 0.04), f.shape), 0, 1)
    return (f * 255).astype(np.uint8), nb


def match(knees: list[dict], gt: np.ndarray) -> list[dict]:
    """Each ground-truth box with its nearest detected knee (by center) and the errors that matter for the crop."""
    rows = []
    for x1, y1, x2, y2 in gt:
        gcx, gcy, gw = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
        d = min(knees, key=lambda k: (k["cx"] - gcx) ** 2 + (k["cy"] - gcy) ** 2, default=None)
        found = d is not None and abs(d["cx"] - gcx) < gw / 2 and abs(d["cy"] - gcy) < (y2 - y1) / 2
        rows.append({"found": bool(found),
                     "center_err": math.hypot(d["cx"] - gcx, d["cy"] - gcy) / gw if found else np.nan,
                     "width_err": d["w"] / gw - 1 if found else np.nan,
                     "crop_iou": iou(crop_box(d["cx"], d["cy"], d["w"]), crop_box(gcx, gcy, gw)) if found else 0.0})
    return rows


def summarize(rows: list[dict], n_images: int, n_extra: int) -> dict:
    df = pd.DataFrame(rows)
    f = df[df.found]
    return {"images": n_images, "knees": len(df), "recall": float(df.found.mean()),
            "extra_detections": int(n_extra),
            "crop_iou_mean": float(df.crop_iou.mean()), "crop_iou_ge_0.8": float((df.crop_iou >= 0.8).mean()),
            "center_err_median": float(f.center_err.median()), "center_err_p95": float(f.center_err.quantile(0.95)),
            "width_err_median_abs": float(f.width_err.abs().median()),
            "width_err_p95_abs": float(f.width_err.abs().quantile(0.95))}


# ---------- model and training (torch) ----------

def build_model(backbone: str = "resnet18", pretrained: bool = True):
    import timm
    import torch
    from torch import nn

    class CenterNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
            chs = self.body.feature_info.channels()
            self.lat = nn.ModuleList(nn.Conv2d(c, 64, 1) for c in chs)
            self.head = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 4, 1))
            self.head[-1].bias.data[0] = -2.19  # heatmap prior ~0.1

        def forward(self, x):
            feats = self.body(x)
            y = self.lat[-1](feats[-1])
            for f, lat in zip(reversed(feats[:-1]), reversed(self.lat[:-1])):
                y = nn.functional.interpolate(y, size=f.shape[-2:], mode="nearest") + lat(f)
            out = self.head(y)  # stride of out_indices[0]
            if out.shape[-2:] != (OUT_H, OUT_W):
                out = nn.functional.interpolate(out, size=(OUT_H, OUT_W), mode="bilinear", align_corners=False)
            return torch.cat([torch.sigmoid(out[:, :1]), out[:, 1:]], 1)

    return CenterNet()


def to_tensor(img: np.ndarray):
    import torch
    t = torch.from_numpy((img.astype(np.float32) / 255 - MEAN) / STD)
    return t[None].repeat(3, 1, 1)


def loss_fn(pred, heat, off, logw, mask):
    import torch
    p = pred[:, 0].clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    neg_w = (1 - heat) ** 4
    focal = -(pos * (1 - p) ** 2 * torch.log(p) + (1 - pos) * neg_w * p ** 2 * torch.log(1 - p)).sum()
    n = pos.sum().clamp(min=1)
    l_off = (torch.abs(pred[:, 1:3] - off) * mask[:, None]).sum() / n
    l_w = (torch.abs(pred[:, 3] - logw) * mask).sum() / n
    return focal / n + l_off + l_w


class DetSet:
    def __init__(self, data: Path, split: str, train: bool, seed: int = 0):
        idx = pd.read_csv(data / "index.csv")
        idx = idx[idx.split == split]
        self.data, self.train = data, train
        self.items = [(g.image.iloc[0], g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32))
                      for _, g in idx.groupby("patient", sort=True)]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def view(self, i):
        path, boxes = self.items[i]
        gray = cv2.imread(str(self.data / path), cv2.IMREAD_GRAYSCALE)
        if self.train:
            return augment(gray, boxes, self.rng)
        img, s, px, py = letterbox(gray)
        nb = boxes.copy()
        nb[:, [0, 2]] = nb[:, [0, 2]] * s + px
        nb[:, [1, 3]] = nb[:, [1, 3]] * s + py
        return img, nb

    def __getitem__(self, i):
        import torch
        img, boxes = self.view(i)
        heat, off, logw, mask = make_targets(boxes)
        return (to_tensor(img), torch.from_numpy(heat), torch.from_numpy(off), torch.from_numpy(logw),
                torch.from_numpy(mask))


def evaluate(model, ds: DetSet, device, k: int = 2) -> tuple[dict, pd.DataFrame]:
    import torch
    model.eval()
    rows, extra, per = [], 0, []
    with torch.no_grad():
        for i in range(len(ds)):
            img, gt = ds.view(i)
            out = model(to_tensor(img)[None].to(device))[0].float().cpu().numpy()
            knees = decode(out[0], out[1:3], out[3], k=k)
            r = match(knees, gt)
            extra += max(0, len(knees) - sum(x["found"] for x in r))
            rows += r
            per += [{"image": ds.items[i][0], **x} for x in r]
    return summarize(rows, len(ds), extra), pd.DataFrame(per)


def main(argv=None):
    import torch
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="outputs/det")
    ap.add_argument("--out", default="runs/detector")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only this many radiographs per split (smoke tests)")
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    sets = {s: DetSet(data, s, train=(s == "train")) for s in ("train", "val", "test")}
    if args.limit:
        for s in sets.values():
            s.items = s.items[:args.limit]
    loader = torch.utils.data.DataLoader(sets["train"], batch_size=args.batch, shuffle=True,
                                         num_workers=args.workers, drop_last=len(sets["train"]) > args.batch,
                                         worker_init_fn=lambda w: setattr(
                                             torch.utils.data.get_worker_info().dataset, "rng",
                                             np.random.default_rng(torch.initial_seed() % 2 ** 32)))
    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * max(1, len(loader)),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, heat, off, logw, mask in loader:
            x, heat, off, logw, mask = (t.to(device) for t in (x, heat, off, logw, mask))
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                pred = model(x)
            loss = loss_fn(pred.float(), heat, off, logw, mask)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        val, _ = evaluate(model, sets["val"], device)
        history.append({"epoch": epoch, "loss": total / max(1, len(loader)), **val})
        print(f"epoch {epoch:3d} loss {history[-1]['loss']:.3f} val recall {val['recall']:.3f} "
              f"crop IoU {val['crop_iou_mean']:.3f} center {val['center_err_median']:.3f} "
              f"width {val['width_err_median_abs']:.3f} ({time.time() - t0:.0f}s)")
        if val["crop_iou_mean"] > best:
            best = val["crop_iou_mean"]
            torch.save({"model": model.state_dict(), "backbone": args.backbone, "epoch": epoch}, out / "best.pt")
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    results = {"best_epoch": ck["epoch"], "crop_factor": CROP_FACTOR}
    for s in ("val", "test"):
        results[s], per = evaluate(model, sets[s], device)
        per.to_csv(out / f"pred_{s}.csv", index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
