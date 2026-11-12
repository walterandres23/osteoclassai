import io
import json
import zipfile

import cv2
import h5py
import numpy as np

from osteoclass.detdata import H5_DIR, CROP_DIR, main, read_h5


def make_h5(img, boxes, classes):
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("images", data=np.repeat(img[..., None], 3, 2).astype(np.float64))
        g = f.create_group("gt_boxes")
        for i, b in enumerate(boxes):
            s = g.create_group(f"i{i}")
            for j, v in enumerate(b):
                s.attrs[f"i{j}"] = float(v)
        c = f.create_group("gt_classes")
        for i, v in enumerate(classes):
            c.attrs[f"i{i}"] = int(v)
    return buf.getvalue()


def test_detdata(tmp_path):
    rng = np.random.default_rng(0)
    zpath = tmp_path / "K.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for k, (pid, folder) in enumerate([("9000001", "trainH5"), ("9000002", "trainH5"), ("9000003", "valH5")]):
            img = cv2.GaussianBlur(rng.integers(0, 255, (256, 320)).astype(np.uint8), (0, 0), 2)
            boxes = [(40, 70, 110, 135), (190, 72, 262, 140)]
            zf.writestr(f"{H5_DIR}{folder}/{pid}.h5", make_h5(img, boxes, [1, 3]))
            if folder == "trainH5":
                # crop of the image-left box, 1.2x larger than the box, as the patient's right knee
                x1, y1, x2, y2 = boxes[0]
                cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, 1.2 * (x2 - x1)
                crop = img[int(cy - s / 2):int(cy + s / 2), int(cx - s / 2):int(cx + s / 2)]
                ok, png = cv2.imencode(".png", cv2.resize(crop, (224, 224)))
                zf.writestr(f"{CROP_DIR}train/1/{pid}R.png", png.tobytes())
    gray, b, c = read_h5(zipfile.ZipFile(zpath).read(f"{H5_DIR}valH5/9000003.h5"))
    assert gray.shape == (256, 320) and b.shape == (2, 4) and list(c) == [1, 3]
    main(["--zip", str(zpath), "--out", str(tmp_path / "det"), "--calibrate", "5"])
    s = json.loads((tmp_path / "det" / "summary.json").read_text())
    assert s["patients"] == 3 and s["boxes"] == 6
    cal = s["calibration"]
    assert cal["knees_score_ge_0.8"] == 2 and cal["share_R_on_image_left"] == 1.0
    assert abs(cal["ratio_w_median"] - 1.2) < 0.05 and abs(cal["dx_median"]) < 0.05
    assert cal["box_cls_equals_kl"] == 1.0 and cal["share_flipped"] == 0.0
