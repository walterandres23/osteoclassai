import numpy as np
from PIL import Image

from osteoclass.data import audit, deduplicate, index_dataset, kl_to_task, patient_split


def _img(path, seed):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rng.integers(0, 255, (32, 32), dtype=np.uint8)).save(path)


def _fake_dataset(root):
    seed = 0
    for p in range(60):
        for knee in (1, 2):
            kl = (p + knee) % 5
            split = "train" if p < 40 else ("val" if p < 48 else "test")
            _img(root / split / str(kl) / f"{9000000 + p}{'L' if knee == 1 else 'R'}.png", seed)
            seed += 1
    # exact duplicate of a train image placed in test (leak)
    src = next((root / "train").rglob("9000000L.png"))
    (root / "test" / src.parent.name).mkdir(parents=True, exist_ok=True)
    (root / "test" / src.parent.name / src.name).write_bytes(src.read_bytes())
    # auto_test copy of a test knee
    t = next((root / "test").rglob("*.png"))
    (root / "auto_test" / t.parent.name).mkdir(parents=True, exist_ok=True)
    (root / "auto_test" / t.parent.name / (t.stem[:-1] + "_1.png")).write_bytes(t.read_bytes())


def test_audit_finds_leaks_and_split_is_clean(tmp_path):
    _fake_dataset(tmp_path)
    df = index_dataset(tmp_path)
    rep = audit(df)
    assert rep["md5_groups_across_splits"] == 1
    assert rep["patients_in_more_than_one_split"] == 1
    assert rep["auto_test_patients_also_in_test"] == 1

    clean = deduplicate(df)
    assert "auto_test" not in set(clean.orig_split)
    assert clean.md5.is_unique

    split = patient_split(clean, seed=0)
    assert split.groupby("patient").split.nunique().max() == 1
    assert set(split.split) == {"train", "val", "test"}


def test_kl_to_task():
    import pandas as pd
    kl = pd.Series([0, 1, 2, 3, 4])
    assert kl_to_task(kl, "oa").tolist() == [0, 0, 1, 1, 1]
    assert kl_to_task(kl, "v1").tolist() == [0, 1, 1, 2, 2]
