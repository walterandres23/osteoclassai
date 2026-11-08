"""Result figures from a run folder (pred_test.csv, history.csv).

    python -m osteoclass.figures --run outputs/runs/convnext_tiny --out media
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 11,
})


def confusion(pred: pd.DataFrame, path: Path) -> None:
    p = pred[[f"p{k}" for k in range(5)]].to_numpy()
    cm = confusion_matrix(pred.kl, p.argmax(1), labels=range(5))
    rows = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            dark = rows[i, j] > 0.55
            ax.text(j, i, f"{rows[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=9,
                    color="white" if dark else INK)
    ax.set_xticks(range(5), [f"KL{k}" for k in range(5)])
    ax.set_yticks(range(5), [f"KL{k}" for k in range(5)])
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("True grade")
    ax.set_title("Confusion matrix, test set (row %)", loc="left", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def roc(pred: pd.DataFrame, path: Path) -> None:
    y = (pred.kl >= 2).astype(int)
    s = pred[["p2", "p3", "p4"]].sum(axis=1)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, ls="--")
    ax.plot(fpr, tpr, color=BLUE, lw=2)
    ax.text(0.55, 0.12, f"AUC {auc:.3f}", color=INK, fontsize=12)
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title("Osteoarthritis (KL ≥ 2), test set", loc="left", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(color=GRID, lw=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def curves(hist: pd.DataFrame, path: Path) -> None:
    best = int(hist.loc[hist.val_qwk.idxmax(), "epoch"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    a.plot(hist.epoch, hist.train_loss, color=BLUE, lw=2)
    a.set_title("Training loss", loc="left", fontsize=12)
    b.plot(hist.epoch, hist.val_qwk, color=BLUE, lw=2, marker="o", ms=4)
    b.set_title("Validation QWK", loc="left", fontsize=12)
    for ax in (a, b):
        ax.axvline(best, color=INK2, lw=1, ls=":")
        ax.set_xlabel("Epoch")
        ax.set_xticks(range(1, int(hist.epoch.max()) + 1, 2))
        ax.grid(color=GRID, lw=0.8)
    lo, hi = hist.val_qwk.min(), hist.val_qwk.max()
    b.set_ylim(lo - 0.02, hi + 0.06)
    b.text(best + 0.3, hi + 0.035, f"best: epoch {best}", color=INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="media")
    a = ap.parse_args(argv)
    run, out = Path(a.run), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pred = pd.read_csv(run / "pred_test.csv")
    confusion(pred, out / "confusion_test.png")
    roc(pred, out / "roc_oa_test.png")
    curves(pd.read_csv(run / "history.csv"), out / "training_curves.png")


if __name__ == "__main__":
    main()
