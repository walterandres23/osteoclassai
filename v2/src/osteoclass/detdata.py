"""Knee-joint detection data from Chen's KneeXrayData (DetKneeData/H5), and how its boxes map to the crops.

Each H5 file (one per patient, deepdish format) holds:
- ``images``: the full bilateral radiograph, downscaled to 256 × 320 × 3;
- ``gt_boxes``: two boxes [x1, y1, x2, y2] in pixels of that image, stored as attributes i0..i3 of groups i0, i1;
- ``gt_classes``: one integer per box (checked here against the KL grade of the matching crop).

``convert`` writes every radiograph as a grayscale PNG plus ``index.csv`` (one row per box).
``calibrate`` finds each KneeKL224 crop inside its radiograph by multi-scale template matching and reports how
the crop relates to the box (side of the image, mirroring, size ratio and center offset), so that a detector can
reproduce the crops the classifiers were trained on.

Usage:
    python -m osteoclass.detdata --zip KneeXrayData.zip --out outputs/det [--calibrate 200]
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd

H5_DIR = "KneeXrayData/DetKneeData/H5/"
CROP_DIR = "KneeXrayData/ClsKLData/kneeKL224/"
SPLITS = {"trainH5": "train", "valH5": "val", "testH5": "test"}


def read_h5(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grayscale uint8 image, boxes (k, 4) as x1, y1, x2, y2, and classes (k,)."""
    with h5py.File(io.BytesIO(data), "r") as f:
        img = np.asarray(f["images"][()], dtype=np.float64)
        gray = np.clip(img.mean(2) if img.ndim == 3 else img, 0, 255).astype(np.uint8)
        groups = f["gt_boxes"]
        keys = sorted(groups.keys(), key=lambda k: int(k[1:]))
        boxes = np.array([[float(groups[k].attrs[f"i{j}"]) for j in range(4)] for k in keys])
        cls = f["gt_classes"].attrs
        classes = np.array([int(cls[f"i{i}"]) for i in range(len(keys))])
    return gray, boxes, classes


def convert(zf: zipfile.ZipFile, out: Path) -> pd.DataFrame:
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for name in sorted(n for n in zf.namelist() if n.startswith(H5_DIR) and n.endswith(".h5")):
        folder, fname = name[len(H5_DIR):].split("/")[:2]
        patient = fname[:-3]
        gray, boxes, classes = read_h5(zf.read(name))
        cv2.imwrite(str(out / "images" / f"{patient}.png"), gray)
        for i, (b, c) in enumerate(zip(boxes, classes)):
            rows.append({"patient": patient, "split": SPLITS[folder], "image": f"images/{patient}.png",
                         "height": gray.shape[0], "width": gray.shape[1], "box": i, "x1": b[0], "y1": b[1],
                         "x2": b[2], "y2": b[3], "cls": int(c)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "index.csv", index=False)
    return df


def find_crop(gray: np.ndarray, crop: np.ndarray, sizes: np.ndarray) -> tuple[float, int, int, int]:
    """Best normalized correlation of ``crop`` (square) inside ``gray`` over the given side lengths."""
    best = (-1.0, 0, 0, 0)
    for s in sizes:
        s = int(round(s))
        if s < 8 or s > min(gray.shape):
            continue
        tpl = cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (float(score), x, y, s)
    return best


def calibrate(zf: zipfile.ZipFile, index: pd.DataFrame, n: int, seed: int = 0) -> dict:
    crops = {}
    for name in zf.namelist():
        if name.startswith(CROP_DIR + "train/") and name.endswith(".png"):
            kl, fname = name.split("/")[-2:]
            crops[fname[:-4]] = (name, int(kl))
    train = index[index.split == "train"]
    patients = train.patient.drop_duplicates().sample(min(n, train.patient.nunique()), random_state=seed)
    rows = []
    for p in patients:
        gray = cv2.imread(str(Path(index.attrs["out"]) / "images" / f"{p}.png"), cv2.IMREAD_GRAYSCALE)
        for side in "LR":
            if f"{p}{side}" not in crops:
                continue
            name, kl = crops[f"{p}{side}"]
            crop = cv2.imdecode(np.frombuffer(zf.read(name), np.uint8), cv2.IMREAD_GRAYSCALE)
            boxes = train[train.patient == p]
            ref = float((boxes.x2 - boxes.x1).mean())
            sizes = np.linspace(0.6, 2.0, 57) * ref
            for flipped in (False, True):
                score, x, y, s = find_crop(gray, crop[:, ::-1].copy() if flipped else crop, sizes)
                cx, cy = x + s / 2, y + s / 2
                b = boxes.iloc[((boxes.x1 + boxes.x2) / 2 - cx).abs().argmin()]
                bw, bh = b.x2 - b.x1, b.y2 - b.y1
                rows.append({"patient": p, "side": side, "kl": kl, "flipped": flipped, "score": score,
                             "crop_size": s, "box": int(b.box), "box_cls": int(b.cls),
                             "box_on_left": bool((b.x1 + b.x2) / 2 < gray.shape[1] / 2),
                             "ratio_w": s / bw, "ratio_h": s / bh, "ratio_max": s / max(bw, bh),
                             "dx": (cx - (b.x1 + b.x2) / 2) / bw, "dy": (cy - (b.y1 + b.y2) / 2) / bh})
    m = pd.DataFrame(rows)
    best = m.loc[m.groupby(["patient", "side"]).score.idxmax()]
    good = best[best.score >= 0.8]
    summary = {
        "knees_matched": int(len(best)), "knees_score_ge_0.8": int(len(good)),
        "median_score": float(best.score.median()),
        "share_flipped": float(good.flipped.mean()),
        "share_L_on_image_left": float(good[good.side == "L"].box_on_left.mean()),
        "share_R_on_image_left": float(good[good.side == "R"].box_on_left.mean()),
        "box_cls_equals_kl": float((good.box_cls == good.kl).mean()),
        **{f"{k}_{q}": float(good[k].quantile(v)) for k in ("ratio_w", "ratio_h", "ratio_max", "dx", "dy")
           for q, v in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))},
    }
    m.to_csv(Path(index.attrs["out"]) / "calibration_matches.csv", index=False)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", default="outputs/det")
    ap.add_argument("--calibrate", type=int, default=200, help="training patients used to relate boxes to crops")
    args = ap.parse_args(argv)
    out = Path(args.out)
    zf = zipfile.ZipFile(args.zip)
    index = convert(zf, out)
    index.attrs["out"] = str(out)
    w, h = index.x2 - index.x1, index.y2 - index.y1
    summary = {"patients": int(index.patient.nunique()), "boxes": int(len(index)),
               "patients_per_split": index.drop_duplicates("patient").split.value_counts().to_dict(),
               "boxes_per_image": index.groupby("patient").size().value_counts().to_dict(),
               "image_sizes": index.drop_duplicates("patient").apply(lambda r: f"{r.height}x{r.width}", axis=1)
               .value_counts().to_dict(),
               "box_w_median": float(w.median()), "box_h_median": float(h.median()),
               "classes": index.cls.value_counts().sort_index().to_dict()}
    if args.calibrate:
        summary["calibration"] = calibrate(zf, index, args.calibrate)
    summary = json.loads(json.dumps(summary, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
