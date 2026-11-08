---
title: OsteoClassAI
emoji: 🦴
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Knee OA grading from X-rays, runs in your browser (ONNX)
---

# OsteoClassAI demo (in-browser)

Upload a posteroanterior knee X-ray cropped to the joint and get the Kellgren-Lawrence grade (0–4), the probability of osteoarthritis (KL ≥ 2) and a class activation map. The model (ConvNeXt-Tiny trained with an ordinal loss, exported to ONNX and quantized to int8) runs in the browser with ONNX Runtime Web, so the image never leaves the visitor's device.

Patient-disjoint test set (1,656 knees): QWK 0.859, 5-grade accuracy 70.1%, AUC 0.951 for KL ≥ 2. Code, data audit and evaluation: <https://github.com/walterandres23/osteoclassai>

Example images: Chen P., *Knee Osteoarthritis Severity Grading Dataset*, Mendeley Data, 2018, doi:10.17632/56rmx5bjcr.1 (from the Osteoarthritis Initiative).

**Research and educational demo. Not a medical device.**
