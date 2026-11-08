import json

import numpy as np
import pandas as pd
from PIL import Image

from osteoclass.metrics import bootstrap_ci, kl5_metrics
from osteoclass.train import main


def test_metrics_perfect():
    y = np.array([0, 1, 2, 3, 4, 2])
    p = np.eye(5)[y]
    m = kl5_metrics(y, p)
    assert m["qwk"] == 1 and m["oa_sensitivity"] == 1 and m["oa_specificity"] == 1
    ci = bootstrap_ci(kl5_metrics, y, p, np.array(list("aabbcc")), n=50)
    assert ci["accuracy"]["value"] == 1


def test_train_smoke(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(60):
        kl, split = i % 5, ("train", "val", "test")[i % 3]
        rel = f"{split}/{kl}/{9000000 + i}L.png"
        (tmp_path / split / str(kl)).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(tmp_path / rel)
        rows.append({"path": rel, "kl": kl, "patient": str(9000000 + i), "split": split})
    pd.DataFrame(rows).to_csv(tmp_path / "splits.csv", index=False)
    out = tmp_path / "run"
    main(["--root", str(tmp_path), "--splits", str(tmp_path / "splits.csv"), "--model", "resnet18",
          "--no-pretrained", "--size", "64", "--epochs", "1", "--batch", "8", "--workers", "0",
          "--bootstrap", "20", "--out", str(out)])
    res = json.loads((out / "results.json").read_text())
    assert "qwk" in res["test"] and (out / "best.pt").exists()
    assert "qwk" in res["test_cuts"] and len(res["expected_grade_cuts"]) == 4

    main(["--root", str(tmp_path), "--splits", str(tmp_path / "splits.csv"), "--model", "resnet18",
          "--no-pretrained", "--size", "64", "--epochs", "1", "--batch", "8", "--workers", "0",
          "--bootstrap", "20", "--loss", "sord", "--tta", "--out", str(tmp_path / "run2")])
    assert (tmp_path / "run2" / "results.json").exists()


def test_sord_and_cuts():
    import torch
    from osteoclass.metrics import decode, tune_cuts
    from osteoclass.train import SORDLoss
    crit = SORDLoss(5, 2.0)
    assert torch.allclose(crit.targets.sum(1), torch.ones(5))
    near = crit(torch.tensor([[0., 0., 5., 0., 0.]]), torch.tensor([3]))
    far = crit(torch.tensor([[5., 0., 0., 0., 0.]]), torch.tensor([3]))
    assert near < far
    y = np.array([0, 1, 2, 3, 4] * 4)
    p = np.eye(5)[y] * 0.8 + 0.04
    assert (decode(p, tune_cuts(y, p, iters=1)) == y).all()
