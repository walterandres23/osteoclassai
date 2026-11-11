import json

import numpy as np
import pandas as pd
import pytest

from osteoclass.ensemble import main


def fake_run(path, seed, noise):
    rng = np.random.default_rng(seed)
    path.mkdir()
    for split, n in (("val", 60), ("test", 80)):
        y = np.arange(n) % 5
        logits = np.eye(5)[y] * 3 + rng.normal(0, noise, (n, 5))
        p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        pd.DataFrame({"path": [f"{split}/{i}.png" for i in range(n)], "patient": np.arange(n) // 2, "kl": y,
                      **{f"p{k}": p[:, k] for k in range(5)}}).to_csv(path / f"pred_{split}.csv", index=False)


def test_ensemble(tmp_path):
    for i, noise in enumerate((2.0, 2.0, 2.5)):
        fake_run(tmp_path / f"r{i}", i, noise)
    main(["--runs", *(str(tmp_path / f"r{i}") for i in range(3)), "--names", "B", "A", "C",
          "--bootstrap", "30", "--out", str(tmp_path / "ens")])
    res = json.loads((tmp_path / "ens" / "results.json").read_text())
    assert res["full"] == "B+A+C" and "+" in res["selected"]
    assert set(res["difference_vs_reference"]) == {"selected", "full"}
    assert len(pd.read_csv(tmp_path / "ens" / "validation.csv")) == 3 + 4


def test_ensemble_rejects_mismatch(tmp_path):
    fake_run(tmp_path / "a", 0, 1.0)
    fake_run(tmp_path / "b", 1, 1.0)
    df = pd.read_csv(tmp_path / "b" / "pred_val.csv")
    df.loc[0, "kl"] = 4
    df.to_csv(tmp_path / "b" / "pred_val.csv", index=False)
    with pytest.raises(ValueError):
        main(["--runs", str(tmp_path / "a"), str(tmp_path / "b"), "--out", str(tmp_path / "e")])
