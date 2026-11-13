import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from osteoclass.calibrate import apply, expected_grade, fit_cuts, fit_shift, main, qwk


def biased(n=400, bias=1.0, seed=0):
    """Probabilities whose expected grade is the true grade plus a bias (over-grading), with noise."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    centre = np.clip(y + bias + rng.normal(0, 0.4, n), 0, 4)
    logits = -((np.arange(5)[None] - centre[:, None]) ** 2) / 0.5
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, p


def test_qwk_matches_sklearn():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 5, 300), rng.integers(0, 5, 300)
    assert abs(qwk(y, p) - cohen_kappa_score(y, p, weights="quadratic")) < 1e-9


def test_fits_recover_bias():
    y, p = biased()
    e = expected_grade(p)
    before = qwk(y, p.argmax(1))
    s = fit_shift(y, e)
    assert 0.6 < s[0] - 0.5 < 1.3                      # shift close to the bias of +1 grade
    assert qwk(y, apply(e, s)) > before + 0.1
    assert qwk(y, apply(e, fit_cuts(y, e))) >= qwk(y, apply(e, s)) - 1e-9


def test_curve_and_fit(tmp_path):
    y, p = biased(300)
    df = pd.DataFrame({"kl": y, **{f"p{k}": p[:, k] for k in range(5)}})
    df.to_csv(tmp_path / "pred.csv", index=False)
    main(["--curve", "--pred", str(tmp_path / "pred.csv"), "--names", "M", "--sizes", "10", "40",
          "--repeats", "5", "--out", str(tmp_path / "cal")])
    curve = pd.read_csv(tmp_path / "cal" / "curve.csv")
    assert set(curve.method) == {"none", "shift", "cuts"} and len(curve) == 2 * 5 * 3
    assert (tmp_path / "cal" / "calibration_curve.png").exists()
    main(["--fit", "--pred", str(tmp_path / "pred.csv"), "--out", str(tmp_path / "cuts.json")])
    res = json.loads((tmp_path / "cuts.json").read_text())
    assert len(res["cuts"]) == 4 and res["fitted_on"]["qwk"] > res["uncalibrated"]["qwk"]
