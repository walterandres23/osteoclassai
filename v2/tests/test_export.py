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
    assert agreement(fp32, int8, [x.numpy()]) == 1
    import onnx
    ops = {n.op_type for n in onnx.load(str(int8)).graph.node}
    assert not ops & {"ConvInteger", "MatMulInteger", "DynamicQuantizeLinear"}


def test_detector_onnx(tmp_path):
    from osteoclass.detect import build_model, IN_H, IN_W
    from osteoclass.export_onnx import detector_agreement, export_detector
    torch.manual_seed(0)
    m = build_model("resnet18", pretrained=False).eval()
    ck = tmp_path / "det.pt"
    torch.save({"model": m.state_dict(), "backbone": "resnet18"}, ck)
    fp32 = export_detector(ck, tmp_path / "detector.onnx")
    int8 = quantize(fp32, tmp_path / "detector_int8.onnx")
    import onnxruntime as ort
    x = torch.randn(1, 3, IN_H, IN_W)
    out = ort.InferenceSession(str(fp32)).run(["det"], {"image": x.numpy()})[0]
    assert out.shape == (1, 4, IN_H // 4, IN_W // 4)
    assert np.allclose(out, m(x).detach().numpy(), atol=1e-4)
    res = detector_agreement(fp32, int8, [np.random.default_rng(0).integers(0, 255, (300, 500), dtype=np.uint8)])
    assert res["same_count"] in (0.0, 1.0)
