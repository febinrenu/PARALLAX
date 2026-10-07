# Validation report

> Decision support only. Not a medical device and not for clinical use. Every number below is on held-out data and carries a 95% confidence interval. This file is generated from `reports/metrics.json` (`make eval`); do not edit it by hand.

## 1. Summary

| Model | Evaluation set | n | Metric | Value [95% CI] | Status |
|---|---|---|---|---|---|
| skin_cls | ISIC 2018 Task 3 official test | 1,512 | balanced multiclass accuracy | 0.680 [0.634, 0.726] | held out |
| brain_cls | leakage-free test split | 563 | accuracy | 0.940 [0.921, 0.958] | held out |
| brain_cls | BDNeuro-MRI, non-duplicate images (external) | 1,644 | accuracy | 0.755 [0.735, 0.775] | external |
| brain_cls (diagnostic model A) | original Kaggle Testing folder | 1,600 | accuracy | 0.942 [0.928, 0.954] | contaminated, reference only |
| brain_seg | LGG held-out patient groups | 9 | mean per-patient Dice | 0.831 [0.716, 0.890] | held out |
| bone_det | FracAtlas test split | 569 | image-level AUROC | 0.923 [0.885, 0.957] | held out |
| chest reader, TorchXRayVision chex | RSNA test (patient-level) | 17,785 | AUROC | 0.785 [0.777, 0.793] | external |
| chest reader, TorchXRayVision mimic_ch | RSNA test (patient-level) | 17,785 | AUROC | 0.749 [0.741, 0.757] | external |
| chest reader, TorchXRayVision all | RSNA test (patient-level) | 17,785 | AUROC | 0.875 [0.869, 0.881] | contaminated, reference only |

What the numbers say, in plain terms:

- **The popular brain benchmark is inflated, and we can measure how much.** The model trained on the original Kaggle split scores 0.942 [0.928, 0.954] on the original Testing folder. Remove the 727 test images that have a near-duplicate in Training and the same model scores 0.903 [0.878, 0.925]; remove also the same-scan neighbours and it scores 0.865 [0.835, 0.894]. On a completely independent set (non-duplicate BDNeuro-MRI images) our leakage-free model scores 0.755 [0.735, 0.775], and pituitary recall is only 0.43.
- **A contaminated model looks much better.** On RSNA the chest model that saw RSNA in training scores AUROC 0.875 [0.869, 0.881]; the two models that never saw it score 0.785 [0.777, 0.793] and 0.749 [0.741, 0.757].
- **Our warnings mean something, with two exceptions.** Instability, abstention and an energy-based out-of-distribution flag all flag cases that are wrong several times as often (section 7). The image-quality gate, with its provisional thresholds, does not predict errors and flags 43% to 91% of clean images; it needs refitting.
- **Calibration works, and conformal sets reach their target** on the official skin test set and the bone and chest false-negative controls; the brain classifier over-covers because its test split is easier than its calibration split (section 5).

## 2. Method

- **Splits.** Group-aware and class-stratified, seed 20261006: HAM10000 by lesion plus duplicate groups, brain MRI by image-similarity clusters (no patient ids exist), LGG by TCGA patient, FracAtlas by duplicate cluster stratified by body part and fracture status, RSNA by duplicate cluster. Official test sets (ISIC 2018 Task 3) are never modified.
- **Leakage audit.** 64-bit DCT perceptual hash; Hamming ≤ 4 is a duplicate. Duplicates are verified against pixel correlation (about 1.0 at distance 0, about 0.93 at distance 4, against 0.12 to 0.47 for random pairs of the same body part or class). Image pairs with different labels at distance ≤ 1 are label noise and both are dropped; pairs with different labels at larger distance are kept as two views of one case and grouped. Flat images get unique pseudo-hashes so blanks never match.
- **Confidence intervals.** 95% percentile cluster bootstrap (B = 1,000 here; B = 2,000 in the training notebooks), resampling whole groups (lesion, patient or similarity cluster), stratified by class where a metric needs every class present; BCa for per-patient Dice with few patients; exact Clopper–Pearson for coverage; subgroup and corruption cells use B = 500 and 300.
- **Calibration.** Temperature scaling on the validation split; randomised-APS split conformal on a separate calibration split (a true top class is always covered, sets are never empty); Platt scaling plus a false-negative-rate threshold for the binary readers; tiers fitted for target accuracies on the calibration split; no fitting on test data.
- **Operating points** (bone sensitivity at 90% specificity) are fixed on the validation split and applied unchanged to the test split.

## 3. Leakage audit

| Dataset | Images | Duplicate pairs (Hamming ≤ 4) | Pairs across original splits | Images removed | Notes |
|---|---|---|---|---|---|
| bdneuro | 5,941 | 287 | 129 | 4,297 | 71% duplicate the Kaggle brain set; only the rest is external test |
| brain_mri | 7,200 | 5,856 | 2,019 | 2,375 | 2,375 removed; loose grouping radius 6 keeps adjacent slices together |
| fracatlas | 4,083 | 341 | 16 | 281 | 59 truncated source files flagged; 2 mis-filed positives |
| ham10000 | 11,527 | 520 | 79 | 296 | 115 training copies of official test images removed; official test untouched |
| lgg_seg | 3,929 | 3,079 | 0 | 0 | slices kept; patients merge only on near-duplicate tumour slices |
| rsna | 26,684 | 1,851 | 0 | 1,278 | patient ids unique per image |

Figure: `reports/figures/brain_leakage.png`.

## 4. External validation and contamination

| Model | Evaluation set | Status | Note |
|---|---|---|---|
| skin_cls | ISIC 2018 Task 3 official test | clean | never used for training, validation or calibration; 115 training copies of test images were removed by the audit |
| skin_cls | HAM10000 internal test split | clean | lesion-grouped split |
| brain_cls | leakage-free test split | clean | similarity-grouped split; no duplicate crosses splits |
| brain_cls | original Kaggle Testing folder | contaminated | 727 of 1,600 images have a near-duplicate in the Kaggle Training folder; shown only to measure inflation |
| brain_cls | BDNeuro-MRI non-duplicate images | clean (external) | 71% of BDNeuro duplicates the Kaggle training source and is excluded; the remaining 1,644 images are independent |
| brain_seg | LGG held-out patient groups | clean | patient-level split; trained on TCGA-LGG only |
| bone_det | FracAtlas test split | clean, patient grouping unavailable | no patient identifiers exist, so another view of a test patient may sit in training |
| xrv densenet121-chex | RSNA test | clean (external) | trained on CheXpert only |
| xrv densenet121-mimic_ch | RSNA test | clean (external) | trained on MIMIC-CXR with CheXpert labels only |
| xrv densenet121-all | RSNA test | contaminated | trained on RSNA among six other datasets; reported for reference and labelled in the UI |
| MedGemma 1.5 4B | FracAtlas, HAM10000, RSNA | unverified | training data overlap with our test sets has not been checked (P3 to confirm from the model card) |
| MedSAM | ISIC 2018 Task 1 | unverified | its training corpus included public skin-lesion data; exact sources not yet checked |

Figures: `reports/figures/cxr_contamination.png`.

**Chest reader localisation** (Grad-CAM++ against RSNA boxes, 1,000 positives): pointing game 0.299 [0.271, 0.326] for chex and 0.196 [0.171, 0.220] for mimic_ch, against a chance level of 0.11; the contaminated weights reach 0.639 [0.609, 0.669]. A heatmap from an externally validated model is therefore weak evidence of where a finding is; the faithfulness test (P1.9) has to decide whether it counts.

## 5. Calibration and conformal prediction

| Model | Eval split | ECE before → after | Coverage at 90% target [95% CI] | Mean set size | Within ±3 points? |
|---|---|---|---|---|---|
| skin_cls | official_test | 0.075 → 0.062 | 0.898 [0.882, 0.913] | 1.26 | yes |
| skin_cls | test | 0.057 → 0.036 | 0.925 [0.907, 0.941] | 1.23 | yes |
| brain_cls | test | 0.079 → 0.019 | 0.940 [0.917, 0.958] | 1.00 | no (over-covers) |
| chest reader (chex), FNR control | RSNA test | ECE 0.220 → 0.018 | miss rate 0.103 (target 0.10) [0.094, 0.112] | n/a | yes |
| bone detector, FNR control | FracAtlas test | ECE 0.061 → 0.035 | miss rate 0.078 (target 0.10) [0.034, 0.147] | n/a | yes |

Figure: `reports/figures/reliability.png`. The chest reader outputs operating-point-scaled scores rather than logits; calibration is Platt scaling on their logit, and only the Lung Opacity and Pneumonia outputs can be calibrated against RSNA.

## 6. Selective prediction

| Reader | AURC [95% CI] | Oracle AURC | Coverage at 5% error | Coverage at 10% error |
|---|---|---|---|---|
| skin_cls (official_test) | 0.059 [0.043, 0.078] | 0.016 | 0.72 | 0.84 |
| brain_cls (test) | 0.014 [0.005, 0.020] | 0.002 | 0.97 | 1.00 |
| chest reader chex (RSNA test) | 0.092 [0.088, 0.097] | 0.027 | 0.39 | 0.59 |
| bone detector (FracAtlas test) | 0.024 [0.014, 0.049] | 0.003 | 0.90 | 1.00 |

Figure: `reports/figures/risk_coverage.png`. AURC is the mean error over all coverage levels when cases are answered most-confident first; the oracle value is the best any ranking could do.

## 7. Trust-signal validation

Error rate when a warning is raised against when it is not. A warning "predicts errors" only if the difference is positive and its confidence interval excludes zero; otherwise it must not be used to downgrade findings.

| Model | Warning | Share raised | Error when raised | Error when not | Difference [95% CI] | Verdict |
|---|---|---|---|---|---|---|
| skin_cls | unstable | 10% | 0.614 | 0.129 | +0.485 [+0.390, +0.581] | predicts errors |
| skin_cls | low_quality | 43% | 0.145 | 0.203 | -0.057 [-0.102, -0.010] | no reliable difference |
| skin_cls | abstain | 7% | 0.622 | 0.143 | +0.479 [+0.356, +0.596] | predicts errors |
| skin_cls | ood_energy | 5% | 0.529 | 0.159 | +0.370 [+0.231, +0.512] | predicts errors |
| brain_cls | unstable | 4% | 0.870 | 0.026 | +0.844 [+0.701, +0.974] | predicts errors |
| brain_cls | low_quality | 88% | 0.061 | 0.059 | +0.002 [-0.057, +0.059] | no reliable difference |
| brain_cls | abstain | 1% | n/a | n/a | n/a | too few flagged or unflagged cases to judge |
| brain_cls | ood_energy | 5% | 0.241 | 0.051 | +0.191 [+0.044, +0.361] | predicts errors |
| cxr_chex | unstable | 2% | n/a | n/a | n/a | too few flagged or unflagged cases to judge |
| cxr_chex | low_quality | 45% | 0.226 | 0.215 | +0.010 [-0.059, +0.082] | no reliable difference |
| cxr_chex | abstain | 17% | 0.524 | 0.159 | +0.365 [+0.250, +0.475] | predicts errors |
| bone_det | unstable | 10% | 0.429 | 0.126 | +0.302 [+0.071, +0.524] | predicts errors |
| bone_det | low_quality | 91% | 0.168 | 0.056 | +0.112 [-0.017, +0.212] | no reliable difference |
| bone_det | abstain | 3% | n/a | n/a | n/a | too few flagged or unflagged cases to judge |

Figure: `reports/figures/trust_signals.png`. Not yet validated because their upstream modules are not finished: `discordant` (needs the complete MedGemma batch from P3; the partial FracAtlas batch shows the second reader finds only about 19% of fractures, so it must not downgrade bone findings) and `unfaithful` (needs P1.9). The harness (`ml/eval/signals.py::compare`) accepts both as soon as a column exists.

### Quality gate on real images (for P1)

| Dataset | Images | Any warning or failure | Failing | Dominant reasons |
|---|---|---|---|---|
| skin_cls | 1000 | 43% | 40% | low_contrast (304), overexposed (93), clipped (28) |
| brain_cls | 563 | 88% | 51% | clipped (472), low_resolution (62), underexposed (50) |
| cxr_chex | 500 | 45% | 14% | clipped (191), rotated (45), blurred (2) |
| bone_det | 203 | 91% | 12% | low_resolution (182), underexposed (17), clipped (10) |

The gate's provisional thresholds, fitted on a synthetic phantom, flag far more than the 5% target on clean images, mainly because black borders count as clipped pixels (`clipped`: brain MRI and chest films), dermoscopy images look low-contrast or overexposed, and small FracAtlas images fall under the 384 px warning. Flagged cases are not wrong more often, so the gate should be refitted on these distributions (the percentile tables are in `reports/signals.json`) before it downgrades anything.

### Out-of-distribution detection (energy score from the specialist's logits)

| Model | OOD set | AUROC, energy score [95% CI] | AUROC, max softmax |
|---|---|---|---|
| skin_cls | cifar10 natural images | 0.864 [0.843, 0.883] | 0.858 |
| skin_cls | brain mri other modality | 0.978 [0.971, 0.984] | 0.982 |
| skin_cls | fracatlas other modality | 0.926 [0.909, 0.942] | 0.884 |
| skin_cls | rsna other modality | 0.895 [0.867, 0.920] | 0.857 |
| brain_cls | cifar10 natural images | 0.826 [0.791, 0.860] | 0.685 |
| brain_cls | ham10000 other modality | 0.828 [0.798, 0.853] | 0.807 |
| brain_cls | fracatlas other modality | 0.877 [0.853, 0.900] | 0.809 |
| brain_cls | rsna other modality | 0.938 [0.923, 0.953] | 0.828 |

Several pairs fall below the 0.95 target; the MedSigLIP Mahalanobis detector (P1.5) is the intended improvement.

## 8. Subgroup audit

Worst subgroup per metric (groups with fewer than 30 images are not reported; missing-metadata rows are shown in `reports/subgroups.json` but never named as worst).

**skin_cls** (ISIC 2018 Task 3 official test, n = 1,512)
- age_band / accuracy: worst **50-69** 0.786 vs 0.826 overall
- age_band / balanced accuracy: worst **50-69** 0.647 vs 0.680 overall
- age_band / melanoma sensitivity: worst **50-69** 0.534 vs 0.637 overall
- sex / accuracy: worst **male** 0.811 vs 0.826 overall
- sex / balanced accuracy: worst **male** 0.642 vs 0.680 overall
- sex / melanoma sensitivity: worst **female** 0.564 vs 0.637 overall
- site / accuracy: worst **Head and neck** 0.738 vs 0.826 overall — confidence interval entirely worse than overall
- site / balanced accuracy: worst **Trunk** 0.601 vs 0.680 overall
- site / melanoma sensitivity: worst **Lower extremity** 0.393 vs 0.637 overall

**brain_cls** (Brain Tumor MRI Dataset, leakage-free test split, n = 563)
- class / recall: worst **meningioma** 0.905 vs 0.940 overall

**bone_det** (FracAtlas test split, n = 569)
- body_part / auroc: worst **leg** 0.870 vs 0.923 overall
- body_part / sensitivity at val threshold: worst **leg** 0.677 vs 0.835 overall
- body_part / specificity at val threshold: worst **mixed** 0.767 vs 0.880 overall
- view / auroc: worst **lateral** 0.891 vs 0.923 overall
- view / sensitivity at val threshold: worst **frontal** 0.836 vs 0.835 overall
- view / specificity at val threshold: worst **oblique** 0.857 vs 0.880 overall
- hardware / auroc: worst **no implant** 0.920 vs 0.923 overall
- hardware / sensitivity at val threshold: worst **no implant** 0.828 vs 0.835 overall
- hardware / specificity at val threshold: worst **no implant** 0.880 vs 0.880 overall

**cxr_chex** (RSNA test split (patient-level), n = 17,785)
- age_band / auroc: worst **70+** 0.733 vs 0.785 overall — confidence interval entirely worse than overall
- age_band / false negative rate at control threshold: worst **<30** 0.163 vs 0.103 overall — confidence interval entirely worse than overall
- sex / auroc: worst **M** 0.778 vs 0.785 overall
- sex / false negative rate at control threshold: worst **M** 0.110 vs 0.103 overall
- view / auroc: worst **AP** 0.699 vs 0.785 overall — confidence interval entirely worse than overall
- view / false negative rate at control threshold: worst **PA** 0.250 vs 0.103 overall — confidence interval entirely worse than overall


Skin-tone labels do not exist in HAM10000 and the brain dataset has no demographic metadata, so those audits are not possible.

## 9. Corruption robustness

| Model | Metric | Clean | Mean of 8 at severity 1 | Mean of 8 at severity 3 | Mean of 8 at severity 5 | Worst single cell |
|---|---|---|---|---|---|---|
| skin_cls | balanced accuracy | 0.685 | 0.691 | 0.644 | 0.543 | gamma/5 = 0.368 |
| brain_cls | balanced accuracy | 0.936 | 0.938 | 0.878 | 0.667 | noise/5 = 0.229 |
| brain_seg | pooled Dice | 0.882 | 0.875 | 0.834 | 0.673 | noise/5 = 0.275 |
| bone_det | AUROC | 0.919 | 0.914 | 0.871 | 0.772 | noise/5 = 0.508 |
| cxr_chex | AUROC | 0.816 | 0.816 | 0.806 | 0.765 | contrast/5 = 0.539 |

Figure: `reports/figures/corruption.png`. Segmentation uses photometric perturbations only (a rotation or crop would move the ground-truth mask too).

## 10. Limitations and open items

- Bone: no patient identifiers, so another view of a test patient may be in training; the bone model has a single dataset and no external test.
- Brain segmentation: one fold of ten (nine test groups), so its interval is wide; the product's single-channel T1-CE inputs are out of the training domain.
- Brain classification: external accuracy (0.755) is far below in-source accuracy; do not present in-source numbers as general performance.
- Skin: no skin-tone audit possible; run-to-run GPU nondeterminism is about 0.01 balanced accuracy.
- Pending: `discordant` and `unfaithful` signal validation (P3, P1.9); zero-shot MedSAM on ISIC 2018 Task 1 (P1.12); MedSigLIP out-of-distribution detector (P1.5); BDNeuro-MRI licence is unconfirmed; overlap of MedGemma and MedSAM training data with our test sets is unverified.
- Reliability and risk-coverage curves for the segmenter are not defined (no per-case probability), so none is reported.

## 11. Reproduce

```
python ml/eval/run_all.py --check     # recompute every number from cached predictions (about 100 s, CPU only)
python ml/eval/figures.py             # redraw reports/figures
python ml/eval/build_cards.py         # model cards and datasheets
python ml/eval/build_report.py        # this file
```
Training and prediction caching need GPUs and the raw datasets (`ml/README.md`); `make eval` does not.
