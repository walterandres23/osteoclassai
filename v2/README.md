# v2 (2026): corrected pipeline

## Data audit

Run `python -m osteoclass.data --root <dataset> --out outputs/data` on the Kaggle dataset (Chen et al., 2019; OAI baseline radiographs):

| Check | Result |
|---|---|
| Images (train / val / test) | 5,778 / 826 / 1,656 = 8,260 |
| Patients | 4,130, two knees each |
| Patients in more than one split | **0** |
| Exact or near duplicates | **0** |
| `auto_test` | Automatic crops of the same 811 test patients, so it is excluded |

The original split is already patient-disjoint, so v2 keeps it. That also makes the results comparable with published work on this dataset. The v1 leakage came from the project itself (doubled images and a random per-image re-split), not from the dataset.

## Baseline: ConvNeXt-Tiny, 5 KL grades

ImageNet-pretrained `convnext_tiny` (timm), 224 × 224, class-weighted cross-entropy with label smoothing, AdamW + one-cycle LR, horizontal flip, ±10° rotation, scale and brightness/contrast augmentation. Model selection on `val` (QWK); `test` evaluated once with the best checkpoint (epoch 9, early stop at 15). Trained on a Colab T4 in about 11 min.

**Test set: 1,656 knees, 828 patients.** 95% CIs from 1,000 bootstrap resamples of patients.

| Metric | Value | 95% CI |
|---|---|---|
| Quadratic weighted kappa | **0.851** | 0.834 – 0.867 |
| Accuracy (5 grades) | 67.3% | 65.0 – 69.8% |
| Macro F1 | 0.714 | 0.689 – 0.737 |
| Mean absolute error (grades) | 0.353 | 0.324 – 0.379 |
| **OA (KL ≥ 2)** sensitivity | 77.5% | 74.3 – 80.7% |
| **OA (KL ≥ 2)** specificity | 94.7% | 93.2 – 96.2% |
| **OA (KL ≥ 2)** ROC AUC | **0.943** | 0.931 – 0.955 |

**Same grouping as v1** (KL0 / KL1–2 / KL3–4), computed from the same predictions: **78.6% accuracy** (95% CI 76.4 – 80.7%), macro F1 0.807. That is on a clean, patient-disjoint test set, and it is already above the 77.1% that v1 reported with leakage.

For reference, Chen et al. (2019), who released this dataset, reported 69.7% accuracy and MAE 0.344 with VGG-19 and an ordinal loss. This baseline is in the same range with plain cross-entropy.

<p align="center">
  <img src="media/confusion_test.png" width="420" alt="Confusion matrix on the test set">
  <img src="media/roc_oa_test.png" width="360" alt="ROC curve for KL >= 2 on the test set">
</p>
<p align="center"><img src="media/training_curves.png" width="700" alt="Training loss and validation QWK per epoch"></p>

Most errors are between neighboring grades, and KL1 ("doubtful") is the hardest class: radiologists also disagree most on it. No KL0 knee was predicted as KL3–4, and no KL4 knee as KL0–2.

## Improvement 1: ordinal loss and test-time flip

KL grades are ordered, so predicting KL3 for a KL4 knee should cost less than predicting KL0. Cross-entropy ignores that. These runs replace it with **SORD** (soft ordinal labels, Díaz & Marathe 2019): the target is a softmax over −α·|k − y|² (α = 2), so neighboring grades get some probability. Predictions average the image and its left-right mirror (TTA). As an option, the expected grade Σ k·p(k) is also mapped to a grade with cut points tuned **on validation only**.

Two runs, same recipe as the baseline otherwise: A at 224 px and B at 288 px. The final model is chosen on validation (QWK with cut points: A 0.852, **B 0.859**); `test` was then evaluated once for each.

**Test set: 1,656 knees, 828 patients**, 95% CIs from 1,000 patient bootstrap resamples.

| Model | Decoding | Accuracy (5 grades) | QWK | MAE | AUC (KL ≥ 2) |
|---|---|---|---|---|---|
| Baseline (CE, 224 px) | argmax | 67.3% | 0.851 (0.834–0.867) | 0.353 | 0.943 |
| A (SORD + TTA, 224 px) | argmax | 70.1% (67.6–72.5) | 0.859 (0.840–0.876) | 0.334 | 0.951 (0.940–0.960) |
| A | cut points | 71.0% | 0.864 (0.846–0.881) | 0.322 | – |
| **B (SORD + TTA, 288 px)** | **argmax** | **70.4% (68.0–72.8)** | **0.867 (0.850–0.884)** | **0.327** | **0.955 (0.945–0.964)** |
| B | cut points | 67.9% | 0.866 | 0.346 | – |
| Chen et al. (2019) | – | 69.7% | – | 0.344 | – |

B with argmax, KL ≥ 2: sensitivity 77.1%, specificity 94.8%. Grouped as in v1 (KL0 / KL1–2 / KL3–4): **81.3%** (95% CI 79.2–83.2%), against 78.6% for the baseline and the 77.1% that v1 reported with leakage.

What this shows:
- The ordinal loss plus TTA is the clear gain: QWK 0.851 → 0.859–0.867 and AUC 0.943 → 0.951–0.955, now above Chen et al. in accuracy and MAE.
- 288 px adds little on top of it: the A and B intervals overlap.
- The cut points helped on validation and for A on test, but not for B (accuracy drops to 67.9%). They are kept as an option, not as the default.

Run: `notebooks/05_ordinal.ipynb` (`python -m osteoclass.train ... --loss sord --tta [--size 288]`).

## Where the model looks (Grad-CAM)

<p align="center"><img src="media/gradcam_test.png" width="900" alt="Grad-CAM on one correctly classified test knee per KL grade"></p>

On one correctly classified test knee per grade, the heat maps sit on the **tibiofemoral joint space**: the medial and lateral margins, where joint-space narrowing and osteophytes show up, and the tibial spines. For KL3–4 the attention concentrates on the most damaged compartment. The model is not relying on text, markers or image borders. These are five examples chosen among correct predictions, so they illustrate the behavior but do not prove it for every image.

## Live demo

**<https://huggingface.co/spaces/walterandres23/osteoclassai>**

<p align="center"><img src="media/demo_web.png" width="760" alt="In-browser demo grading a KL2 knee"></p>

The demo runs **model A** (224 px): it is almost as good as B and keeps the page fast. Like the evaluation, it averages the image and its mirror. The model is exported to ONNX (`osteoclass.export_onnx`) with two outputs: the logits and a class activation map (the classifier applied at every location of the last feature map, so no gradients are needed). Weights are stored as int8 per output channel and dequantized on load: 111 MB → 28 MB, with **100% agreement** with the float model on 300 test images. The page (`web/index.html`) runs it with ONNX Runtime Web in about 2 s on a laptop CPU, and the X-ray never leaves the device. Hosted as a free static Hugging Face Space (`notebooks/04_publicar_demo.ipynb`).

`app/` holds an equivalent Gradio version with Grad-CAM, for running locally (`python app/app.py` with `best.pt` next to it).

## Next

1. External validation on another cohort (for example MOST) to check how the model generalizes beyond the OAI.

## How to run

`notebooks/02_entrenamiento.ipynb` (Colab, GPU): clones this repo, downloads the dataset with `kagglehub`, runs the audit and the training, and copies the results to Google Drive.

Locally: `pip install -e .` then `python -m osteoclass.data ...` and `python -m osteoclass.train --root <dataset> --splits outputs/data/splits.csv`. Figures: `python -m osteoclass.figures --run <run folder> --out media`. Tests: `pytest -q tests`.

## References

Chen P, Gao L, Shi X, Allen K, Yang L. Fully automatic knee osteoarthritis severity grading using deep neural networks with a novel ordinal loss. *Computerized Medical Imaging and Graphics* 75 (2019) 84–92. doi:10.1016/j.compmedimag.2019.06.002

Díaz R, Marathe A. Soft labels for ordinal regression. *CVPR* 2019, 4738–4747.
