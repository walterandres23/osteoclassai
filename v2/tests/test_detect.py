import json

import cv2
import numpy as np
import pandas as pd
import pytest

from osteoclass.detect import (IN_H, IN_W, augment, crop_box, decode, letterbox, make_targets, match, to_original)

BOXES = np.array([[40, 70, 106, 135], [190, 72, 256, 138]], np.float32)


def radiograph(rng=None):
    img = np.full((IN_H, IN_W), 40, np.uint8)
    for x1, y1, x2, y2 in BOXES.astype(int):
        img[y1:y2, x1:x2] = 220
    return img


def test_targets_roundtrip():
    heat, off, logw, mask = make_targets(BOXES)
    assert mask.sum() == 2 and heat.max() == 1.0
    knees = decode(heat, off, logw)
    assert len(knees) == 2
    for k, (x1, y1, x2, y2) in zip(knees, BOXES):
        assert abs(k["cx"] - (x1 + x2) / 2) < 1e-3 and abs(k["cy"] - (y1 + y2) / 2) < 1e-3
        assert abs(k["w"] - (x2 - x1)) < 1e-3
    rows = match(knees, BOXES)
    assert all(r["found"] and r["crop_iou"] > 0.999 for r in rows)


def test_letterbox_back_to_original():
    gray = np.zeros((600, 400), np.uint8)
    img, s, px, py = letterbox(gray)
    assert img.shape == (IN_H, IN_W)
    k = to_original({"cx": px + 100 * s, "cy": py + 300 * s, "w": 66 * s, "score": 1}, s, px, py)
    assert abs(k["cx"] - 100) < 1e-6 and abs(k["cy"] - 300) < 1e-6 and abs(k["w"] - 66) < 1e-6
    assert crop_box(100, 100, 50) == (67.5, 67.5, 132.5, 132.5)


def test_augment_keeps_boxes_on_the_knees():
    rng = np.random.default_rng(0)
    for _ in range(60):
        img, nb = augment(radiograph(), BOXES, rng)
        assert img.shape == (IN_H, IN_W) and img.dtype == np.uint8
        for x1, y1, x2, y2 in nb:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            inside = img[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3].mean()
            assert inside > img.mean(), "box center should fall on the bright knee"


def test_detector_smoke(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    from osteoclass.detect import main
    (tmp_path / "det" / "images").mkdir(parents=True)
    rows = []
    for i, split in enumerate(["train"] * 4 + ["val"] * 2 + ["test"] * 2):
        cv2.imwrite(str(tmp_path / "det" / "images" / f"{i}.png"), radiograph())
        for b, (x1, y1, x2, y2) in enumerate(BOXES):
            rows.append({"patient": i, "split": split, "image": f"images/{i}.png", "height": IN_H, "width": IN_W,
                         "box": b, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "cls": 0})
    pd.DataFrame(rows).to_csv(tmp_path / "det" / "index.csv", index=False)
    main(["--data", str(tmp_path / "det"), "--out", str(tmp_path / "run"), "--epochs", "1", "--batch", "2",
          "--workers", "0", "--no-pretrained"])
    res = json.loads((tmp_path / "run" / "results.json").read_text())
    assert set(res) >= {"val", "test", "best_epoch"} and res["test"]["knees"] == 4
