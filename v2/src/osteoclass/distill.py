"""Teacher for distillation: the ensemble of trained runs, applied to the training images.

Each run predicts every training image at its own input size, with the mirror average (TTA) it was evaluated
with; the teacher probability is the equal-weight mean over runs, as in ``osteoclass.ensemble``. The student is
then trained with ``osteoclass.train --teacher <this CSV>``.

Note: the runs saw these images during training, so their probabilities here are more confident than on new
images (the usual caveat of distilling on the training set).

Usage:
    python -m osteoclass.distill --root <dataset> --splits outputs/data/splits.csv \\
        --runs <A> <B> <C> <D> --out outputs/teacher_train.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from .dataset import KneeDataset, build_transforms
from .train import predict


def teacher_probs(runs: list[str | Path], df: pd.DataFrame, root, device, batch: int = 64, workers: int = 2):
    probs, y = [], None
    for run in runs:
        run = Path(run)
        ck = torch.load(run / "best.pt", map_location=device)
        model = timm.create_model(ck["model"], pretrained=False, num_classes=5)
        model.load_state_dict(ck["state_dict"])
        model.to(device).eval()
        tta = False
        if (run / "results.json").exists():
            tta = bool(json.loads((run / "results.json").read_text()).get("config", {}).get("tta", False))
        ds = KneeDataset(df, root, "kl5", build_transforms(int(ck.get("size", 224)), train=False))
        yy, p = predict(model, DataLoader(ds, batch_size=batch, num_workers=workers), device, tta)
        y = yy if y is None else y
        probs.append(p)
        print(f"{run.name}: size {ck.get('size', 224)}, tta {tta}, train accuracy {(p.argmax(1) == yy).mean():.3f}",
              flush=True)
        del model
    return y, np.mean(probs, 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="outputs/teacher_train.csv")
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(args.splits)
    df = df[df.split == args.split].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    y, p = teacher_probs(args.runs, df, args.root, device, workers=args.workers)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"path": df.path, "kl": y, **{f"p{k}": p[:, k] for k in range(5)}}).to_csv(out, index=False)
    print(f"teacher on {len(df)} {args.split} images: accuracy {(p.argmax(1) == y).mean():.3f}, "
          f"mean max probability {p.max(1).mean():.3f} -> {out}")


if __name__ == "__main__":
    main()
