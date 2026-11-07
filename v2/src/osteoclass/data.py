"""Dataset indexing, auditing and patient-level splitting.

Expected layout (Kaggle "Knee Osteoarthritis Dataset with Severity Grading"):

    <root>/<split>/<KL grade 0-4>/<patient>_<knee>.png

where <split> is any subfolder (train, val, test, auto_test...). File names such as
``9003175_1.png`` carry the Osteoarthritis Initiative patient ID and the knee.

Usage:
    python -m osteoclass.data --root data/kneeKL --out outputs/data
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# 9003175L.png / 9003175R.png (train, val, test) or 9003175_1.png (auto_test)
FILE_RE = re.compile(r"^(?P<patient>\d+)(?:_(?P<num>\d)|(?P<side>[LR]))")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}
# auto_test holds automatic crops of the same knees as test: never train on it.
DERIVED_SPLITS = {"auto_test"}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dhash(path: Path, size: int = 16) -> str:
    """Difference hash: identical for the same image re-encoded or slightly resized."""
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def index_dataset(root: str | Path) -> pd.DataFrame:
    """One row per image: path, original split, KL grade, patient, knee and hashes."""
    root = Path(root)
    rows = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for kl_dir in sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.isdigit()):
            for img in sorted(kl_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                m = FILE_RE.match(img.stem)
                rows.append({
                    "path": str(img.relative_to(root)),
                    "orig_split": split_dir.name,
                    "kl": int(kl_dir.name),
                    "patient": m["patient"] if m else None,
                    "knee": (m["side"] or m["num"]) if m else None,
                    "md5": _md5(img),
                    "dhash": _dhash(img),
                })
    if not rows:
        raise FileNotFoundError(f"No images found under {root}/<split>/<0-4>/")
    return pd.DataFrame(rows)


def audit(df: pd.DataFrame) -> dict:
    """Counts, duplicates and patient overlap between the original splits."""
    base = df[~df.orig_split.isin(DERIVED_SPLITS)]
    report: dict = {
        "images_per_split": df.orig_split.value_counts().sort_index().to_dict(),
        "kl_per_split": {s: g.kl.value_counts().sort_index().to_dict()
                         for s, g in df.groupby("orig_split")},
        "patients": int(base.patient.nunique()),
        "images_without_patient_id": int(base.patient.isna().sum()),
    }

    for key in ("md5", "dhash"):
        groups = base.groupby(key)
        dup = groups.filter(lambda g: len(g) > 1)
        report[f"{key}_duplicate_groups"] = int(dup[key].nunique())
        report[f"{key}_duplicate_images"] = int(len(dup))
        cross = dup.groupby(key).orig_split.nunique()
        report[f"{key}_groups_across_splits"] = int((cross > 1).sum())
        conflicting = dup.groupby(key).kl.nunique()
        report[f"{key}_groups_with_conflicting_kl"] = int((conflicting > 1).sum())

    # Same patient in more than one original split.
    per_patient = base.dropna(subset=["patient"]).groupby("patient").orig_split.nunique()
    report["patients_in_more_than_one_split"] = int((per_patient > 1).sum())
    splits = sorted(base.orig_split.unique())
    pats = {s: set(base.loc[base.orig_split == s, "patient"].dropna()) for s in splits}
    report["patient_overlap"] = {f"{a}&{b}": len(pats[a] & pats[b])
                                 for i, a in enumerate(splits) for b in splits[i + 1:]}
    # Same knee (patient + side) more than once.
    knees = base.dropna(subset=["patient"]).groupby(["patient", "knee"]).size()
    report["knees_with_more_than_one_image"] = int((knees > 1).sum())

    # Is auto_test the same knees as test?
    if "auto_test" in df.orig_split.values and "test" in df.orig_split.values:
        a = set(df.loc[df.orig_split == "auto_test", "patient"].dropna())
        t = set(df.loc[df.orig_split == "test", "patient"].dropna())
        report["auto_test_patients_also_in_test"] = len(a & t)
        report["auto_test_patients"] = len(a)
    return report


def deduplicate(df: pd.DataFrame, key: str = "md5") -> pd.DataFrame:
    """Drop derived splits and keep one image per duplicate group (exact copies by default).

    Perceptual hashes are reported by ``audit`` but not used to drop images: knee
    X-rays look alike and a coarse hash merges different knees.

    Groups whose copies disagree on the KL grade are dropped entirely (label noise).
    """
    base = df[~df.orig_split.isin(DERIVED_SPLITS)].copy()
    kl_n = base.groupby(key).kl.transform("nunique")
    base = base[kl_n == 1]
    return base.drop_duplicates(subset=key, keep="first").reset_index(drop=True)


def patient_split(df: pd.DataFrame, test_frac: float = 0.15, val_frac: float = 0.15,
                  seed: int = 42) -> pd.DataFrame:
    """Assign train/val/test so that every patient falls in exactly one split.

    Stratified by the patient's highest KL grade to keep the class balance.
    """
    df = df.copy()
    df["group"] = df.patient.fillna(pd.Series("img_" + df.index.astype(str), index=df.index))
    strat = df.groupby("group").kl.transform("max")

    def carve(frame, strat_col, frac):
        n_splits = max(2, round(1 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, hold = next(sgkf.split(frame, strat_col, groups=frame.group))
        mask = np.zeros(len(frame), bool)
        mask[hold] = True
        return mask

    test_mask = carve(df, strat, test_frac)
    df["split"] = None
    df.loc[test_mask, "split"] = "test"
    rest = df[~test_mask]
    val_mask = carve(rest, strat[~test_mask], val_frac / (1 - test_frac))
    df.loc[rest.index[val_mask], "split"] = "val"
    df["split"] = df["split"].fillna("train")

    assert df.groupby("group").split.nunique().max() == 1, "patient leaked across splits"
    return df.drop(columns="group")


def kl_to_task(kl: pd.Series, task: str) -> pd.Series:
    """Label for each task: 'kl5' (0-4), 'oa' (KL >= 2 -> 1) or 'v1' (0 / 1-2 / 3-4)."""
    if task == "kl5":
        return kl
    if task == "oa":
        return (kl >= 2).astype(int)
    if task == "v1":
        return kl.map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2})
    raise ValueError(task)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="folder with <split>/<0-4>/ images")
    ap.add_argument("--out", default="outputs/data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=["original", "patient"], default="original",
                    help="keep the dataset's own split (patient-disjoint) or make a new patient-level one")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = index_dataset(args.root)
    df.to_csv(out / "index.csv", index=False)

    rep = audit(df)
    clean = deduplicate(df)
    if args.split == "original":
        # The Kaggle split is already patient-disjoint (checked in the audit).
        assert rep["patients_in_more_than_one_split"] == 0, "original split leaks patients"
        split = clean.assign(split=clean.orig_split)
    else:
        split = patient_split(clean, seed=args.seed)
    rep["after_dedup_images"] = int(len(clean))
    rep["new_split_images"] = split.split.value_counts().to_dict()
    rep["new_split_patients"] = split.groupby("split").patient.nunique().to_dict()
    rep["new_split_kl"] = {s: g.kl.value_counts().sort_index().to_dict() for s, g in split.groupby("split")}

    split.to_csv(out / "splits.csv", index=False)
    (out / "audit.json").write_text(json.dumps(rep, indent=2, default=int))
    print(json.dumps(rep, indent=2, default=int))


if __name__ == "__main__":
    main()
