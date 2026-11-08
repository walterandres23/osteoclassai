import numpy as np
import timm
import torch

from osteoclass.export_onnx import CAMWrapper, agreement, export, quantize


def test_cam_wrapper_and_onnx(tmp_path):
    torch.manual_seed(0)
    m = timm.create_model("convnext_atto", pretrained=False, num_classes=5).eval()
    w = CAMWrapper(m).eval()
    x = torch.randn(1, 3, 64, 64)
    logits, cam = w(x)
    assert torch.allclose(logits, m(x), atol=1e-5)
    assert cam.shape[:2] == (1, 5)

    ck = tmp_path / "best.pt"
    torch.save({"model": "convnext_atto", "size": 64, "state_dict": m.state_dict()}, ck)
    fp32 = export(ck, tmp_path / "model.onnx")
    int8 = quantize(fp32, tmp_path / "model_int8.onnx")
    assert int8.stat().st_size < fp32.stat().st_size

    import onnxruntime as ort
    out = ort.InferenceSession(str(fp32)).run(None, {"image": x.numpy()})
    assert np.allclose(out[0], logits.detach().numpy(), atol=1e-4)
    assert 0 <= agreement(fp32, int8, [x.numpy()]) <= 1
