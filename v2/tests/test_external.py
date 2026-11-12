import json

import numpy as np
import timm
import torch
from PIL import Image

from osteoclass.external import audit, joint_box, load_gray, main, pad_square

NAMES = ["0Normal", "1Doubtful", "2Mild", "3Moderate", "4Severe"]


def _knee(rng, w=300, h=162):
    return rng.integers(0, 255, (h, w), dtype=np.uint8)


def _save(root, expert, grade, name, arr):
    d = root / expert / NAMES[grade]
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(d / name)


def _fake_dataset(root):
    rng = np.random.default_rng(0)
    imgs = {f"k{i}": _knee(rng) for i in range(10)}
    imgs["bil"] = _knee(rng, 640, 161)
    imgs["bil2"] = _knee(rng, 640, 161)
    grades = {f"k{i}": i % 5 for i in range(10)} | {"bil": 3, "bil2": 1}
    for ex in ("MedicalExpert-I", "MedicalExpert-II"):
        for k, a in imgs.items():
            g = grades[k]
            if ex == "MedicalExpert-II" and k == "k0":
                g = 1                                       # experts disagree on k0
            _save(root, ex, g, f"{k}.png", a)
        _save(root, ex, grades["k2"], "k2_copy.png", imgs["k2"])               # exact copy
        _save(root, ex, 4, "k3_conflict.png", np.clip(imgs["k3"].astype(int) + 3, 0, 255).astype(np.uint8))
    return grades


def test_audit(tmp_path):
    _fake_dataset(tmp_path)
    df, s = audit(tmp_path)
    assert s["files_per_expert"]["MedicalExpert-I"] == 14
    assert s["unique_images"] == 13 and s["exact_copies_removed"] == 1
    assert s["images_experts_disagree"] == 1
    assert s["clusters_with_conflicting_grades"] == 1           # k3 and its brighter copy, grades 3 vs 4
    assert s["analysis_knees_bilateral"] == 2
    assert s["analysis_knees_single"] == 8                      # k0..k9 minus k0 (disagree) and k3 (conflict)
    assert df.representative.sum() == 10


def test_load_and_pad(tmp_path):
    a = np.linspace(0, 65535, 300 * 162).astype(np.uint16).reshape(162, 300)
    Image.fromarray(a).save(tmp_path / "x16.png")
    g = np.asarray(load_gray(tmp_path / "x16.png"))
    assert g.max() == 255 and 120 < g.mean() < 135            # scaled to 8 bits, not clipped to white
    im = Image.fromarray(g)
    sq = pad_square(im)
    assert sq.size == (300, 300) and np.asarray(sq)[0].max() == 0


def test_joint_box():
    a = np.full((162, 300), 20, np.uint8)
    a[:, 120:220] = 200                       # knee: bright columns 120-219
    a[100:106, 120:220] = 90                  # joint line at rows 100-105
    x, y, s = joint_box(Image.fromarray(a))
    assert abs((x + s / 2) - 170) <= 2 and abs((y + s / 2) - 102) <= 4 and abs(s - 111) <= 2


def test_external_smoke(tmp_path):
    root = tmp_path / "data"
    _fake_dataset(root)
    run = tmp_path / "run"
    run.mkdir()
    m = timm.create_model("resnet18", pretrained=False, num_classes=5)
    torch.save({"model": "resnet18", "size": 64, "state_dict": m.state_dict()}, run / "best.pt")
    (run / "results.json").write_text(json.dumps({"config": {"tta": True}}))
    out = tmp_path / "out"
    main(["--root", str(root), "--runs", str(run), "--out", str(out), "--bootstrap", "20",
          "--exploratory"])
    res = json.loads((out / "results.json").read_text())
    r = res["models"]["run"]
    assert r["tta"] and r["single_pad"]["n"] == 8 and r["bilateral_pad"]["n"] == 2
    assert "qwk" in r["single_stretch"] and (out / "audit.csv").exists()
    assert "qwk" in r["exploratory_single_joint"] and "qwk" in r["exploratory_recalibrated_pad"]


def test_external_detector_smoke(tmp_path):
    from osteoclass.detect import build_model
    root = tmp_path / "data"
    _fake_dataset(root)
    run = tmp_path / "run"
    run.mkdir()
    m = timm.create_model("resnet18", pretrained=False, num_classes=5)
    torch.save({"model": "resnet18", "size": 64, "state_dict": m.state_dict()}, run / "best.pt")
    det = tmp_path / "det.pt"
    torch.save({"model": build_model("resnet18", pretrained=False).state_dict(), "backbone": "resnet18"}, det)
    out = tmp_path / "out"
    main(["--root", str(root), "--runs", str(run), "--out", str(out), "--bootstrap", "20", "--detector", str(det)])
    res = json.loads((out / "results.json").read_text())
    r = res["models"]["run"]
    assert res["detector"]["single_total"] == 8 and res["detector"]["bilateral_total"] == 2
    assert "qwk" in r["exploratory_single_detector"] and "qwk" in r["exploratory_recalibrated_detector"]
    assert r["exploratory_bilateral_detector"]["n"] == 2 and (out / "detections_single.csv").exists()
