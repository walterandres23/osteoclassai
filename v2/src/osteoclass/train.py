"""Train and evaluate a KL-grading model.

Usage (Colab, GPU):
    python -m osteoclass.train --root <dataset> --splits outputs/data/splits.csv \
        --model convnext_tiny --epochs 25 --out outputs/runs/convnext_tiny

Model selection uses only the validation split. The test split is evaluated once,
at the end, with the best checkpoint.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import KneeDataset, build_transforms
from .metrics import bootstrap_ci, kl5_metrics, tune_cuts


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=n_classes).astype(float)
    w = counts.sum() / (n_classes * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


class SORDLoss(nn.Module):
    """Soft ordinal labels (Diaz & Marathe, CVPR 2019): neighbors of the true grade get some target
    mass, so predicting KL3 for a KL2 knee costs less than predicting KL0."""

    def __init__(self, n_classes: int = 5, alpha: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        ranks = torch.arange(n_classes, dtype=torch.float32)
        self.register_buffer("targets", torch.softmax(-alpha * (ranks[None] - ranks[:, None]) ** 2, 1))
        self.weight = weight

    def forward(self, logits, y):
        loss = -(self.targets[y] * torch.log_softmax(logits.float(), 1)).sum(1)
        if self.weight is not None:
            w = self.weight.to(loss.device)[y]
            return (loss * w).sum() / w.sum()
        return loss.mean()


@torch.no_grad()
def predict(model, loader, device, tta: bool = False) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.softmax(model(x).float(), 1)
            if tta:  # left/right mirror
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--model", default="convnext_tiny")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images per split")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--loss", choices=["ce", "sord"], default="ce")
    ap.add_argument("--sord-alpha", type=float, default=2.0)
    ap.add_argument("--tta", action="store_true", help="average with the horizontally flipped image")
    ap.add_argument("--out", default="outputs/runs/run")
    args = ap.parse_args(argv)

    torch.manual_seed(42)
    np.random.seed(42)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.splits)
    parts = {s: df[df.split == s] for s in ("train", "val", "test")}
    if args.limit:
        parts = {s: d.sample(min(args.limit, len(d)), random_state=0) for s, d in parts.items()}

    tf_train = build_transforms(args.size, train=True)
    tf_eval = build_transforms(args.size, train=False)
    ds = {s: KneeDataset(d, args.root, "kl5", tf_train if s == "train" else tf_eval) for s, d in parts.items()}
    dl = {s: DataLoader(ds[s], batch_size=args.batch, shuffle=(s == "train"), num_workers=args.workers,
                        pin_memory=device.type == "cuda") for s in ds}

    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=5).to(device)
    cw = class_weights(ds["train"].labels, 5).to(device)
    crit = (SORDLoss(5, args.sord_alpha, cw).to(device) if args.loss == "sord"
            else nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=max(1, len(dl["train"])), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best, bad, history = -np.inf, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in dl["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        yv, pv = predict(model, dl["val"], device, args.tta)
        mv = kl5_metrics(yv, pv)
        row = {"epoch": epoch, "train_loss": total / n, **{f"val_{k}": v for k, v in mv.items()},
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        print(f"epoch {epoch:02d} loss {row['train_loss']:.3f} val_qwk {mv['qwk']:.3f} "
              f"val_acc {mv['accuracy']:.3f} val_oa_auc {mv['oa_auc']:.3f} ({row['seconds']}s)", flush=True)
        if mv["qwk"] > best:
            best, bad = mv["qwk"], 0
            torch.save({"model": args.model, "size": args.size, "state_dict": model.state_dict()}, out / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)

    # Final evaluation: best checkpoint, test split, once.
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["state_dict"])
    results = {}
    preds = {s: predict(model, dl[s], device, args.tta) for s in ("val", "test")}
    cuts = tune_cuts(*preds["val"])          # fitted on validation only
    results["expected_grade_cuts"] = cuts
    for s in ("val", "test"):
        y, p = preds[s]
        groups = parts[s].patient.astype(str).to_numpy()
        results[s] = bootstrap_ci(kl5_metrics, y, p, groups, n=args.bootstrap)
        results[f"{s}_cuts"] = bootstrap_ci(lambda a, b: kl5_metrics(a, b, cuts), y, p, groups, n=args.bootstrap)
        pd.DataFrame({"path": parts[s].path.to_numpy(), "patient": groups, "kl": y,
                      **{f"p{k}": p[:, k] for k in range(5)}}).to_csv(out / f"pred_{s}.csv", index=False)
    results["config"] = vars(args)
    (out / "results.json").write_text(json.dumps(results, indent=2))
    t = results["test"]
    print(f"TEST  qwk {t['qwk']['value']:.3f} [{t['qwk']['ci_low']:.3f}-{t['qwk']['ci_high']:.3f}]  "
          f"acc {t['accuracy']['value']:.3f}  OA(KL>=2) sens {t['oa_sensitivity']['value']:.3f} "
          f"spec {t['oa_specificity']['value']:.3f} auc {t['oa_auc']['value']:.3f}")
    c = results["test_cuts"]
    print(f"TEST (expected grade, val-tuned cuts {[round(v, 2) for v in cuts]})  qwk {c['qwk']['value']:.3f} "
          f"[{c['qwk']['ci_low']:.3f}-{c['qwk']['ci_high']:.3f}]  acc {c['accuracy']['value']:.3f}  "
          f"mae {c['mae_grade']['value']:.3f}")


if __name__ == "__main__":
    main()
