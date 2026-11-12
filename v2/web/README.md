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

Upload a posteroanterior knee X-ray, a full radiograph with one or both knees or a crop of the joint, and get the Kellgren-Lawrence grade (0–4) of each knee, the probability of osteoarthritis (KL ≥ 2) and a class activation map. A knee-joint detector (CenterNet on ResNet-18, trained on Chen's detection data) finds each knee and crops it at the scale the classifier was trained on; the classifier is ConvNeXt-Tiny with an ordinal loss and robustness augmentation. Both are exported to ONNX, quantized to int8 and run in the browser with ONNX Runtime Web, so the image never leaves the visitor's device.

Patient-disjoint test set (1,656 knees): QWK 0.867, 5-grade accuracy 70.2%, AUC 0.950 for KL ≥ 2. On X-rays from another hospital the detector crop recovers most of the lost agreement (QWK 0.65 instead of 0.57 on Gornale & Patravali), but grades tend to be higher than the local experts'; see the external test in the repository. Code, data audit and evaluation: <https://github.com/walterandres23/osteoclassai>

Example images: Chen P., *Knee Osteoarthritis Severity Grading Dataset*, Mendeley Data, 2018, doi:10.17632/56rmx5bjcr.1 (from the Osteoarthritis Initiative). Full radiographs: Gornale S. S., Patravali P. U., *Digital Knee X-ray Images*, Mendeley Data, 2020, doi:10.17632/t9ndx37v5h.1 (CC BY 4.0).

**Research and educational demo. Not a medical device.**
