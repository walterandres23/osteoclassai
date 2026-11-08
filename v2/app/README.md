---
title: OsteoClassAI
emoji: 🦴
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Knee osteoarthritis KL grading from X-rays + Grad-CAM
---

# OsteoClassAI demo

Upload a posteroanterior knee X-ray cropped to the joint and get the Kellgren-Lawrence grade (0–4), the probability of osteoarthritis (KL ≥ 2) and a Grad-CAM heat map.

- Model: ConvNeXt-Tiny fine-tuned on the Knee Osteoarthritis Severity Grading dataset (Chen et al., 2019, OAI).
- Test set (patient-disjoint, 1,656 knees): QWK 0.851, 5-grade accuracy 67.3%, AUC 0.943 for KL ≥ 2.
- Code, data audit and evaluation: <https://github.com/walterandres23/osteoclassai>

Example images come from the test split of that dataset: Chen P., *Knee Osteoarthritis Severity Grading Dataset*, Mendeley Data, 2018, doi:10.17632/56rmx5bjcr.1, organized from the Osteoarthritis Initiative (OAI).

**Research and educational demo. Not a medical device.**
