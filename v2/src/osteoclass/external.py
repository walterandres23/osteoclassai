"""External test on the Digital Knee X-ray Images dataset (Gornale & Patravali, Mendeley Data 2020,
doi:10.17632/t9ndx37v5h.1, CC BY 4.0): 1650 knee radiographs from hospitals in India, graded by two experts.

The protocol was fixed before looking at any prediction:

1. Audit. Both expert folders hold the same images, so images are matched by pixel hash. Exact copies
   are merged, and near-duplicates (the same knee with another exposure or contrast, perceptual hash
   distance <= 50 of 256 bits) are grouped into clusters, one cluster per knee.
2. Labels: expert I. Images where the two experts disagree are dropped, and so are clusters whose
   images carry different grades.
3. Unit of analysis: one image per cluster (the first by file name); bootstrap CIs resample clusters.
4. Main analysis: single-knee images (300 x 162 px). Bilateral images (640 x 161 px, one grade for the
   pair) are a secondary analysis: each half is graded and the higher expected grade is kept.
5. Preprocessing: pad to a square, keeping the aspect ratio, then the v2 eval transform. Sensitivity
   analysis: plain resize (stretched). 16-bit images are scaled to 8 bits.
6. Models are used as trained (frozen), with the same decoding and TTA as in their own evaluation.

Exploratory analyses (--exploratory), added AFTER seeing the main results and reported separately:
the main results showed a systematic over-grading with the order of grades preserved, and the padded
knees are about half the scale of the training crops. So (a) "joint": a square crop centered on the
joint whose side is the knee width / 0.9, the field of view of the Chen et al. crops; and (b) split-half
recalibration: expected-grade cut points fitted on half of the knees, scored on the other half
(20 random splits by cluster, both directions).

Usage:
    python -m osteoclass.external --root "<...>/Knee X-ray Images" \
        --runs <run_dir> [<run_dir> ...] --out outputs/external/gornale
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .dataset import build_transforms
from .metrics import bootstrap_ci, kl5_metrics

EXPERTS = ("MedicalExpert-I", "MedicalExpert-II")
NEAR_DUP_BITS = 50          # of 256 (16 x 16 pHash); checked by eye: up to ~48 is the same knee
BILATERAL_MIN_RATIO = 3.0   # width / height; single knees are ~1.85, bilateral ~4.0


def load_gray(path: str | Path) -> Image.Image:
    """8-bit grayscale. 16-bit PNGs are scaled by 1/257 (full range), not clipped."""
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im).astype(np.float32) / 257.0
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")
    return im.convert("L")


def pad_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new("L", (s, s), 0)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def joint_box(img: Image.Image, fill: float = 0.9) -> tuple[int, int, int]:
    """Square box (x, y, side) centered on the knee joint, without training: the knee is the widest run of
    bright (bone) columns, the joint line the darkest row band between 25 % and 75 % of the height."""
    from scipy.ndimage import uniform_filter1d
    a = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = a.shape
    col = uniform_filter1d(a.mean(0), 9)
    lo, hi = np.percentile(col, 5), col.max()
    idx = np.flatnonzero(col > lo + 0.6 * (hi - lo))
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 3) + 1)
    run = max(runs, key=len)
    x0, x1 = int(run[0]), int(run[-1])
    row = uniform_filter1d(a[:, x0:x1 + 1].mean(1), 5)
    b0, b1 = int(h * 0.25), int(h * 0.75)
    cy = b0 + int(np.argmin(row[b0:b1]))
    side = max(8, int(round((x1 - x0 + 1) / fill)))
    return int((x0 + x1) / 2 - side / 2), int(cy - side / 2), side


def crop_joint(img: Image.Image) -> Image.Image:
    x, y, s = joint_box(img)
    out = Image.new("L", (s, s), 0)
    out.paste(img.convert("L"), (-x, -y))
    return out


def phash(img: Image.Image, size: int = 16) -> np.ndarray:
    """256-bit DCT perceptual hash as a boolean vector."""
    import imagehash
    return imagehash.phash(img, hash_size=size).hash.flatten()


def audit(root: str | Path) -> tuple[pd.DataFrame, dict]:
    """One row per unique image (pixel hash) with both experts' grades, flags and near-duplicate cluster."""
    root = Path(root)
    recs = []
    for ex in EXPERTS:
        for p in sorted((root / ex).glob("*/*.png")):
            img = load_gray(p)
            recs.append({"expert": ex, "path": str(p.relative_to(root)).replace("\\", "/"),
                         "grade": int(p.parent.name[0]),
                         "md5": hashlib.md5(np.asarray(img).tobytes()).hexdigest(),
                         "width": img.size[0], "height": img.size[1]})
    raw = pd.DataFrame(recs)
    if raw.empty:
        raise FileNotFoundError(f"no PNG files under {root}/{EXPERTS[0]}/<grade>/")

    rows = []
    for md5, g in raw.groupby("md5", sort=False):
        e1, e2 = g[g.expert == EXPERTS[0]], g[g.expert == EXPERTS[1]]
        grades1, grades2 = set(e1.grade), set(e2.grade)
        first = (e1 if len(e1) else g).sort_values("path").iloc[0]
        rows.append({"path": first.path, "md5": md5, "width": first.width, "height": first.height,
                     "copies": len(e1), "grade_I": min(grades1) if grades1 else -1,
                     "grade_II": min(grades2) if grades2 else -1,
                     "copy_conflict": len(grades1) > 1 or len(grades2) > 1})
    df = pd.DataFrame(rows).sort_values("path").reset_index(drop=True)
    df["experts_disagree"] = df.grade_I != df.grade_II
    df["bilateral"] = df.width / df.height >= BILATERAL_MIN_RATIO

    # near-duplicate clusters (union-find on pHash Hamming distance)
    H = np.stack([phash(load_gray(root / p)) for p in df.path])
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(df)):
        close = np.flatnonzero((H[i + 1:] != H[i]).sum(1) <= NEAR_DUP_BITS) + i + 1
        for j in close:
            parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(df))]
    ids = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    df["cluster"] = [ids[r] for r in roots]

    kept = df[~df.experts_disagree & ~df.copy_conflict]
    n_grades = kept.groupby("cluster").grade_I.nunique()
    df["cluster_conflict"] = df.cluster.map(n_grades).fillna(1) > 1
    df["usable"] = ~df.experts_disagree & ~df.copy_conflict & ~df.cluster_conflict
    df["representative"] = False
    rep = df[df.usable].groupby("cluster").head(1).index
    df.loc[rep, "representative"] = True

    both = df[(df.grade_I >= 0) & (df.grade_II >= 0)]
    summary = {
        "files_per_expert": {ex: int((raw.expert == ex).sum()) for ex in EXPERTS},
        "unique_images": len(df),
        "exact_copies_removed": int(len(raw[raw.expert == EXPERTS[0]]) - len(df)),
        "near_duplicate_clusters": int(df.cluster.nunique()),
        "experts_exact_agreement": float((both.grade_I == both.grade_II).mean()),
        "experts_qwk": float(cohen_kappa_score(both.grade_I, both.grade_II, weights="quadratic")),
        "images_experts_disagree": int(df.experts_disagree.sum()),
        "images_copy_conflict": int(df.copy_conflict.sum()),
        "clusters_with_conflicting_grades": int(df[df.cluster_conflict].cluster.nunique()),
        "analysis_knees_single": int((df.representative & ~df.bilateral).sum()),
        "analysis_knees_bilateral": int((df.representative & df.bilateral).sum()),
        "grades_single": {int(k): int(v) for k, v in
                          df[df.representative & ~df.bilateral].grade_I.value_counts().sort_index().items()},
    }
    return df, summary


def load_run(run_dir: str | Path, device) -> tuple[torch.nn.Module, int, bool, str]:
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / "best.pt", map_location=device)
    model = timm.create_model(ck["model"], pretrained=False, num_classes=5)
    model.load_state_dict(ck["state_dict"])
    tta = False
    if (run_dir / "results.json").exists():
        tta = bool(json.loads((run_dir / "results.json").read_text()).get("config", {}).get("tta", False))
    return model.to(device).eval(), int(ck.get("size", 224)), tta, run_dir.name


@torch.no_grad()
def predict_images(model, images: list[Image.Image], size: int, tta: bool, device, batch: int = 64) -> np.ndarray:
    tf = build_transforms(size, train=False)
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([tf(im) for im in images[i:i + batch]]).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.softmax(model(x).float(), 1)
            if tta:
                p = (p + torch.softmax(model(torch.flip(x, dims=[3])).float(), 1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 5))


def prepare(img: Image.Image, mode: str) -> Image.Image:
    if mode == "pad":
        return pad_square(img)
    if mode == "joint":
        return crop_joint(img)
    return img


def split_half_recalibration(y, prob, groups, n_splits: int = 20) -> dict:
    """Fit expected-grade cut points on half of the clusters, score the other half; both directions."""
    from .metrics import decode, tune_cuts
    g3 = lambda k: np.select([k == 0, k <= 2], [0, 1], 2)
    rows = []
    for seed in range(n_splits):
        cl = np.unique(groups)
        np.random.default_rng(seed).shuffle(cl)
        a = np.isin(groups, cl[: len(cl) // 2])
        for fit, ev in ((a, ~a), (~a, a)):
            cuts = tune_cuts(y[fit], prob[fit])
            m = kl5_metrics(y[ev], prob[ev], cuts)
            pred = decode(prob[ev], cuts)
            rows.append((m["qwk"], m["accuracy"], m["mae_grade"], float((g3(pred) == g3(y[ev])).mean())))
    r = np.array(rows)
    return {k: {"mean": float(r[:, i].mean()), "min": float(r[:, i].min()), "max": float(r[:, i].max())}
            for i, k in enumerate(("qwk", "accuracy", "mae_grade", "accuracy_3class"))}


def three_class_accuracy(y, prob) -> float:
    g = lambda k: np.select([k == 0, k <= 2], [0, 1], 2)
    return float((g(prob.argmax(1)) == g(y)).mean())


def evaluate(y, prob, groups, n_boot: int) -> dict:
    res = bootstrap_ci(kl5_metrics, y, prob, groups, n=n_boot)
    pred = prob.argmax(1)
    res["accuracy_3class"] = three_class_accuracy(y, prob)
    res["within_one_grade"] = float((np.abs(pred - y) <= 1).mean())
    res["per_grade_accuracy"] = {int(k): float((pred[y == k] == k).mean()) for k in np.unique(y)}
    res["confusion"] = confusion_matrix(y, pred, labels=list(range(5))).tolist()
    res["n"] = int(len(y))
    from scipy.stats import spearmanr
    res["spearman_expected_grade"] = float(spearmanr(prob @ np.arange(prob.shape[1]), y).correlation)
    res["mean_expected_grade_by_true_grade"] = {int(k): float((prob[y == k] @ np.arange(5)).mean())
                                                for k in np.unique(y)}
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder that contains MedicalExpert-I and MedicalExpert-II")
    ap.add_argument("--runs", nargs="+", required=True, help="run folders with best.pt (and results.json)")
    ap.add_argument("--out", default="outputs/external/gornale")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--exploratory", action="store_true",
                    help="also run the post-hoc analyses: joint-centered crop and split-half recalibration")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, summary = audit(root)
    df.to_csv(out / "audit.csv", index=False)
    (out / "audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    single = df[df.representative & ~df.bilateral].reset_index(drop=True)
    bilat = df[df.representative & df.bilateral].reset_index(drop=True)
    imgs_single = [load_gray(root / p) for p in single.path]
    halves = []
    for p in bilat.path:
        im = load_gray(root / p)
        w, h = im.size
        halves += [im.crop((0, 0, w // 2, h)), im.crop((w // 2, 0, w, h))]

    results = {"protocol": __doc__.split("Usage:")[0].strip(), "audit": summary, "models": {}}
    for run in args.runs:
        model, size, tta, name = load_run(run, device)
        r = {"size": size, "tta": tta}
        modes = ("pad", "stretch", "joint") if args.exploratory else ("pad", "stretch")
        for mode in modes:
            p = predict_images(model, [prepare(im, mode) for im in imgs_single], size, tta, device)
            y = single.grade_I.to_numpy()
            key = f"exploratory_single_{mode}" if mode == "joint" else f"single_{mode}"
            r[key] = evaluate(y, p, single.cluster.to_numpy(), args.bootstrap)
            if args.exploratory and mode in ("pad", "joint"):
                r[f"exploratory_recalibrated_{mode}"] = split_half_recalibration(y, p, single.cluster.to_numpy())
            pd.DataFrame({"path": single.path, "cluster": single.cluster, "kl": y,
                          **{f"p{k}": p[:, k] for k in range(5)}}).to_csv(out / f"pred_{name}_{mode}.csv", index=False)
        if len(bilat):
            ph = predict_images(model, [prepare(im, "pad") for im in halves], size, tta, device).reshape(-1, 2, 5)
            worse = (ph @ np.arange(5)).argmax(1)              # half with the higher expected grade
            pb = ph[np.arange(len(ph)), worse]
            r["bilateral_pad"] = evaluate(bilat.grade_I.to_numpy(), pb, bilat.cluster.to_numpy(), args.bootstrap)
        results["models"][name] = r
        s = r["single_pad"]
        print(f"{name:28s} single/pad  qwk {s['qwk']['value']:.3f} [{s['qwk']['ci_low']:.3f}-{s['qwk']['ci_high']:.3f}]"
              f"  acc {s['accuracy']['value']:.3f}  mae {s['mae_grade']['value']:.3f}"
              f"  auc(KL>=2) {s['oa_auc']['value']:.3f}  3-class {s['accuracy_3class']:.3f}"
              f"  | stretch qwk {r['single_stretch']['qwk']['value']:.3f}", flush=True)
        if args.exploratory:
            j, rp, rj = r["exploratory_single_joint"], r["exploratory_recalibrated_pad"], r["exploratory_recalibrated_joint"]
            print(f"{'':28s} EXPLORATORY joint crop qwk {j['qwk']['value']:.3f} [{j['qwk']['ci_low']:.3f}-"
                  f"{j['qwk']['ci_high']:.3f}]  acc {j['accuracy']['value']:.3f}  auc(KL>=2) {j['oa_auc']['value']:.3f}"
                  f"  | recalibrated qwk: pad {rp['qwk']['mean']:.3f}  joint {rj['qwk']['mean']:.3f}", flush=True)
        del model
    (out / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
