"""Ensembles of finished runs, from the predictions each run already saved (no GPU needed).

Protocol, fixed before looking at any ensemble on the test set:
- Candidates: every subset of two or more of the given runs, averaging their class probabilities with equal
  weights (each run's probabilities already include its mirror TTA when it was trained with --tta).
- Decoding: expected grade with four cut points fitted on validation only, as for the single models.
- Selection: highest validation QWK. Ties go to the smaller ensemble, then to the order of the runs given.
- Test: evaluated once for the selected ensemble (main result) and for the ensemble of all runs (pre-declared
  sensitivity analysis). Both are compared with the reference run (the first one given, B in the README) by a
  paired bootstrap over patients.

Usage:
    python -m osteoclass.ensemble --runs <B> <A> <C> <D> --names B A C D --out outputs/ensemble
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci, kl5_metrics, tune_cuts

PCOLS = [f"p{k}" for k in range(5)]
REPORT = ("qwk", "accuracy", "mae_grade", "oa_auc")


def load_run(run: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(Path(run) / f"pred_{split}.csv")
    return df.sort_values("path").reset_index(drop=True)


def aligned(runs: list[Path], split: str):
    """Labels, patients and an array (runs, n, 5) of probabilities, checked to refer to the same images."""
    frames = [load_run(r, split) for r in runs]
    ref = frames[0]
    for r, f in zip(runs[1:], frames[1:]):
        if len(f) != len(ref) or not (f.path.to_numpy() == ref.path.to_numpy()).all() \
                or not (f.kl.to_numpy() == ref.kl.to_numpy()).all():
            raise ValueError(f"{r}: pred_{split}.csv does not cover the same images as {runs[0]}")
    probs = np.stack([f[PCOLS].to_numpy(float) for f in frames])
    return ref.kl.to_numpy(int), ref.patient.astype(str).to_numpy(), probs


def paired_bootstrap(y, pa, pb, groups, cuts_a, cuts_b, n: int = 2000, seed: int = 0) -> dict:
    """Difference b − a for each reported metric, resampling patients; point value and 95% CI."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    point = {k: kl5_metrics(y, pb, cuts_b)[k] - kl5_metrics(y, pa, cuts_a)[k] for k in REPORT}
    samples = {k: [] for k in REPORT}
    for _ in range(n):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        try:
            ma, mb = kl5_metrics(y[idx], pa[idx], cuts_a), kl5_metrics(y[idx], pb[idx], cuts_b)
        except ValueError:
            continue
        for k in REPORT:
            samples[k].append(mb[k] - ma[k])
    return {k: {"value": float(point[k]), "ci_low": float(np.nanpercentile(samples[k], 2.5)),
                "ci_high": float(np.nanpercentile(samples[k], 97.5))} for k in REPORT}


def candidates(n_runs: int):
    for size in range(2, n_runs + 1):
        yield from itertools.combinations(range(n_runs), size)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run folders; the first is the reference")
    ap.add_argument("--names", nargs="+", help="short names, same order as --runs")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--out", default="outputs/ensemble")
    args = ap.parse_args(argv)
    runs = [Path(r) for r in args.runs]
    names = args.names or [r.name for r in runs]
    if len(names) != len(runs) or len(runs) < 2:
        ap.error("give at least two runs and one name per run")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    yv, _, pv = aligned(runs, "val")
    # Validation only: single runs and every ensemble.
    table = []
    for combo in [(i,) for i in range(len(runs))] + list(candidates(len(runs))):
        p = pv[list(combo)].mean(0)
        cuts = tune_cuts(yv, p)
        table.append({"members": "+".join(names[i] for i in combo), "size": len(combo), "combo": combo,
                      "cuts": cuts, "val_qwk": kl5_metrics(yv, p, cuts)["qwk"]})
    val = pd.DataFrame(table)
    ens = val[val["size"] >= 2]
    best = ens.loc[ens.sort_values(["val_qwk", "size"], ascending=[False, True], kind="stable").index[0]]
    full = ens[ens["size"] == len(runs)].iloc[0]
    val.drop(columns=["combo"]).sort_values("val_qwk", ascending=False).to_csv(out / "validation.csv", index=False)
    print(val.drop(columns=["combo", "cuts"]).sort_values("val_qwk", ascending=False).to_string(index=False))
    print(f"\nselected on validation: {best['members']} (val QWK {best['val_qwk']:.3f})")

    # Test, once, for the reference run, the selected ensemble and the full ensemble.
    yt, gt, pt = aligned(runs, "test")
    ref = val.iloc[0]
    results = {"protocol": __doc__.split("Usage:")[0].strip(), "reference": names[0],
               "selected": best["members"], "full": full["members"], "test": {}, "difference_vs_reference": {}}
    p_ref = pt[0]
    for label, row in (("reference", ref), ("selected", best), ("full", full)):
        p = pt[list(row["combo"])].mean(0)
        results["test"][label] = {"members": row["members"], "cuts": row["cuts"], "val_qwk": row["val_qwk"],
                                  **bootstrap_ci(lambda a, b: kl5_metrics(a, b, row["cuts"]), yt, p, gt,
                                                 n=args.bootstrap)}
        if label != "reference":
            results["difference_vs_reference"][label] = paired_bootstrap(
                yt, p_ref, p, gt, ref["cuts"], row["cuts"], n=args.bootstrap)
            pd.DataFrame({"path": load_run(runs[0], "test").path, "patient": gt, "kl": yt,
                          **{c: p[:, k] for k, c in enumerate(PCOLS)}}).to_csv(out / f"pred_test_{label}.csv",
                                                                               index=False)
    (out / "results.json").write_text(json.dumps(results, indent=2))

    for label in ("reference", "selected", "full"):
        t = results["test"][label]
        line = "  ".join(f"{k} {t[k]['value']:.3f} [{t[k]['ci_low']:.3f}-{t[k]['ci_high']:.3f}]" for k in REPORT)
        print(f"TEST {label:9s} {t['members']:10s} {line}")
    for label, d in results["difference_vs_reference"].items():
        line = "  ".join(f"{k} {d[k]['value']:+.3f} [{d[k]['ci_low']:+.3f} to {d[k]['ci_high']:+.3f}]"
                         for k in REPORT)
        print(f"DIFF {label} − {names[0]}: {line}")


if __name__ == "__main__":
    main()
