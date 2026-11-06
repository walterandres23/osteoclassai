# v2 (2026): corrected pipeline (in progress)

Goals, in order:

1. **Clean data:** remove duplicate images and split **by patient** (OAI ID in the file name), keeping a locked test set that is only used once at the end.
2. **Baseline on original radiographs** (no Sobel), with class weights or focal loss.
3. **Proper metrics:** 5-class accuracy and quadratic weighted kappa, plus the binary task KL ≥ 2 with sensitivity, specificity and ROC AUC, all with 95% bootstrap confidence intervals at patient level.
4. **Explainability:** Grad-CAM to check that the model looks at the joint space.
5. **Demo:** small web app to upload an X-ray and see the grade and the heat map.

Training runs on Google Colab (GPU). The code lives in this repository as Python modules; the notebook only clones the repo, downloads the dataset with the Kaggle API, and calls them.
