import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image

from osteoclass.gradcam import GradCAM, grid, pick_examples


def test_gradcam_shapes_and_grid(tmp_path):
    torch.manual_seed(0)
    model = timm.create_model("resnet18", pretrained=False, num_classes=5)
    heat, probs = GradCAM(model)(torch.randn(1, 3, 64, 64))
    assert heat.shape == (64, 64) and 0 <= heat.min() and heat.max() <= 1
    assert abs(probs.sum() - 1) < 1e-5

    rows = []
    rng = np.random.default_rng(0)
    for k in range(5):
        rel = f"test/{k}/900000{k}L.png"
        (tmp_path / "test" / str(k)).mkdir(parents=True)
        Image.fromarray(rng.integers(0, 255, (64, 64), dtype=np.uint8)).save(tmp_path / rel)
        rows.append({"path": rel, "kl": k, **{f"p{j}": float(j == k) for j in range(5)}})
    ex = pick_examples(pd.DataFrame(rows))
    assert len(ex) == 5
    grid(model, 64, tmp_path, ex, tmp_path / "g.png")
    assert (tmp_path / "g.png").exists()
