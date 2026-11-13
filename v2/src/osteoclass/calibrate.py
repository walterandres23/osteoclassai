"""Local calibration: adapt the grade cut points to a new site with a few knees graded by its own experts.

On another hospital the models keep the order of the grades but grade higher (external test). The grade is read
from the expected grade E = sum_k k·p_k cut at four points; calibrating means choosing those points with a small
labeled sample from the new site. Two ways, compared here:

- ``shift``: one parameter, the four default cuts (0.5, 1.5, 2.5, 3.5) moved together by d;
- ``cuts``: the four cuts fitted freely (coordinate search on a 0.025 grid, as on validation).

Learning curve (``--curve``): for each sample size n, draw n knees at random (one image per knee), fit on them and
score the remaining knees of the site; repeat many times. Uncalibrated (argmax) is scored on the same remainder.
Deployment (``--fit``): fit on all the labeled knees given and write the cuts to a JSON file.

Usage:
    python -m osteoclass.calibrate --curve --pred pred_B.csv pred_D.csv --names B D --out outputs/calibration
    python -m osteoclass.calibrate --fit --pred my_site_predictions.csv --method shift --out cuts.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PCOLS = [f"p{k}" for k in range(5)]
DEFAULT_CUTS = np.array([0.5, 1.5, 2.5, 3.5])
GRID = np.linspace(0, 4, 161)
SHIFTS = np.round(np.arange(-2.0, 2.0001, 0.02), 2)
_W = (np.subtract.outer(np.arange(5), np.arange(5)) ** 2) / 16.0


def qwk(y: np.ndarray, pred: np.ndarray) -> float:
    """Quadratic weighted kappa for 5 grades (same value as sklearn, much faster)."""
    o = np.bincount(y * 5 + pred, minlength=25).reshape(5, 5).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / max(o.sum(), 1)
    den = (_W * e).sum()
    return float(1 - (_W * o).sum() / den) if den > 0 else 0.0


def expected_grade(prob: np.ndarray) -> np.ndarray:
    return prob @ np.arange(prob.shape[1])


def apply(e: np.ndarray, cuts) -> np.ndarray:
    return np.searchsorted(np.asarray(cuts), e)


def fit_shift(y: np.ndarray, e: np.ndarray) -> list[float]:
    """Best common shift d of the default cuts; ties go to the smallest |d|."""
    scores = np.array([qwk(y, apply(e, DEFAULT_CUTS + d)) for d in SHIFTS])
    best = np.flatnonzero(scores >= scores.max() - 1e-12)
    d = SHIFTS[best[np.argmin(np.abs(SHIFTS[best]))]]
    return (DEFAULT_CUTS + d).tolist()


def fit_cuts(y: np.ndarray, e: np.ndarray, iters: int = 3) -> list[float]:
    """Four cuts by coordinate search, starting from the default ones (as metrics.tune_cuts, faster)."""
    cuts = DEFAULT_CUTS.tolist()
    for _ in range(iters):
        for i in range(4):
            lo = cuts[i - 1] + 0.01 if i else 0.0
            hi = cuts[i + 1] - 0.01 if i < 3 else 4.0
            cand = GRID[(GRID >= lo) & (GRID <= hi)]
            if not len(cand):
                continue
            scores = [qwk(y, apply(e, cuts[:i] + [c] + cuts[i + 1:])) for c in cand]
            best = np.flatnonzero(np.array(scores) >= max(scores) - 1e-12)
            cuts[i] = float(cand[best[np.argmin(np.abs(cand[best] - cuts[i]))]])  # ties: stay close
    return cuts


FITTERS = {"shift": fit_shift, "cuts": fit_cuts}


def scores(y, pred) -> dict:
    return {"qwk": qwk(y, pred), "accuracy": float((pred == y).mean()), "mae": float(np.abs(pred - y).mean())}


def learning_curve(y: np.ndarray, prob: np.ndarray, sizes, repeats: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    e, argmax = expected_grade(prob), prob.argmax(1)
    rows = []
    for n in sizes:
        if n >= len(y):
            continue
        for r in range(repeats):
            fit = np.zeros(len(y), bool)
            fit[rng.choice(len(y), n, replace=False)] = True
            ev = ~fit
            rows.append({"n": n, "repeat": r, "method": "none", **scores(y[ev], argmax[ev])})
            for name, fitter in FITTERS.items():
                cuts = fitter(y[fit], e[fit])
                rows.append({"n": n, "repeat": r, "method": name, **scores(y[ev], apply(e[ev], cuts))})
    return pd.DataFrame(rows)


def summarize(curve: pd.DataFrame) -> list[dict]:
    out = []
    for (model, n, method), g in curve.groupby(["model", "n", "method"], sort=False):
        row = {"model": model, "n": int(n), "method": method}
        for k in ("qwk", "accuracy", "mae"):
            row.update({f"{k}_median": float(g[k].median()), f"{k}_p10": float(g[k].quantile(0.1)),
                        f"{k}_p90": float(g[k].quantile(0.9))})
        out.append(row)
    return out


def plot(curve: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(dict.fromkeys(curve.model))
    fig, axes = plt.subplots(1, len(models), figsize=(5.2 * len(models), 3.8), sharey=True, squeeze=False)
    colors = {"none": "#8a8a8a", "shift": "#2a78d6", "cuts": "#d6762a"}
    labels = {"none": "uncalibrated (argmax)", "shift": "shift of the cuts (1 parameter)",
              "cuts": "4 cut points"}
    for ax, model in zip(axes[0], models):
        for method in ("none", "shift", "cuts"):
            g = curve[(curve.model == model) & (curve.method == method)].groupby("n").qwk
            med, lo, hi = g.median(), g.quantile(0.1), g.quantile(0.9)
            ax.plot(med.index, med.values, "o-", color=colors[method], label=labels[method], ms=4)
            ax.fill_between(med.index, lo.values, hi.values, color=colors[method], alpha=0.15)
        ax.set_xscale("log")
        ax.set_xticks(sorted(curve.n.unique()))
        ax.set_xticklabels(sorted(curve.n.unique()))
        ax.set_title(model)
        ax.set_xlabel("knees graded at the new site")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("QWK on the other knees of the site")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def load(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    return df.kl.to_numpy(int), df[PCOLS].to_numpy(float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--curve", action="store_true", help="learning curve over the sample size")
    mode.add_argument("--fit", action="store_true", help="fit cuts on all the knees given and save them")
    ap.add_argument("--pred", nargs="+", required=True, help="CSV with columns kl, p0..p4 (one row per knee)")
    ap.add_argument("--names", nargs="+")
    ap.add_argument("--sizes", nargs="+", type=int, default=[10, 20, 40, 80, 160, 320])
    ap.add_argument("--repeats", type=int, default=200)
    ap.add_argument("--method", choices=list(FITTERS), default="shift")
    ap.add_argument("--out", default="outputs/calibration")
    args = ap.parse_args(argv)

    if args.fit:
        y, prob = load(args.pred[0])
        cuts = FITTERS[args.method](y, expected_grade(prob))
        res = {"method": args.method, "knees": int(len(y)), "cuts": cuts,
               "fitted_on": scores(y, apply(expected_grade(prob), cuts)),
               "uncalibrated": scores(y, prob.argmax(1))}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    names = args.names or [Path(p).stem for p in args.pred]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    curves = []
    for path, name in zip(args.pred, names):
        y, prob = load(path)
        c = learning_curve(y, prob, args.sizes, args.repeats)
        c.insert(0, "model", name)
        curves.append(c)
    curve = pd.concat(curves, ignore_index=True)
    curve.to_csv(out / "curve.csv", index=False)
    summary = summarize(curve)
    (out / "summary.json").write_text(json.dumps({"protocol": __doc__.split("Usage:")[0].strip(),
                                                  "repeats": args.repeats, "rows": summary}, indent=2))
    plot(curve, out / "calibration_curve.png")
    table = pd.DataFrame(summary)
    for model in names:
        t = table[table.model == model].pivot(index="n", columns="method", values="qwk_median")
        print(f"\n{model} · median QWK on the rest of the site ({args.repeats} draws per size)")
        print(t[["none", "shift", "cuts"]].round(3).to_string())


if __name__ == "__main__":
    main()
