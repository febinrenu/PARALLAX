# Model card: Skin lesion classifier

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Registry:** `skin_cls@a217517b` · weights sha256 `1fb2ea5ab200743b…` · split `9ef391891e85…` · commit `a217517b`
**Task:** dermoscopy lesion classification into 7 classes (akiec, bcc, bkl, df, mel, nv, vasc), with a localisation mask from MedSAM downstream.
**Architecture:** ConvNeXt-Tiny (timm `convnext_tiny.fb_in22k_ft_in1k`), 224 px, fine-tuned for 15 epochs (AdamW, label smoothing 0.1, √-inverse-frequency class sampling, GPU augmentation).
**Licence:** Apache-2.0 backbone; fine-tuned on HAM10000 (CC BY-NC 4.0), so the weights are for non-commercial use.

## Intended use and out-of-scope use
A second opinion on a single dermoscopic image, shown to a doctor with its evidence. Not for clinical photographs, not for non-dermoscopic images, not for lesion types outside the seven classes, and never as a standalone diagnosis.

## Training data
HAM10000 / ISIC 2018 Task 3 training set (10,015 images, 7,470 lesions). Split by lesion (and duplicate group) into train / validation / calibration / internal test (70/10/10/10). The leakage audit (pHash, Hamming ≤ 4) found 520 duplicate pairs and removed 296 images, including 115 training copies of official test images. See the datasheet.

## Evaluation
Official ISIC 2018 Task 3 test set (1,512 images, never used in training, validation or calibration). Confidence intervals are 95% cluster bootstrap over lesions (lesion ids from the ISIC API), resampled within class.
- **Balanced multiclass accuracy (official metric): 0.680 (95% CI 0.634 to 0.726)**
- accuracy 0.826 (95% CI 0.805 to 0.847); macro-F1 0.722 (95% CI 0.676 to 0.763); macro AUROC 0.953 (95% CI 0.936 to 0.967)
- melanoma: AUROC 0.924 (95% CI 0.898 to 0.949); sensitivity (recall) 0.637 (95% CI 0.544 to 0.728)
- per-class recall: akiec 0.42, bcc 0.65, bkl 0.79, df 0.75, mel 0.64, nv 0.92, vasc 0.60
A second identical run scored 0.689, so run-to-run GPU nondeterminism is about 0.01.

## Calibration and abstention
- temperature T = 0.923 (fitted on the validation split); ECE 0.075 → 0.062 on the test split
- split-conformal (randomised APS, fitted on the calibration split), target coverage 90%: empirical coverage 0.898 (exact 95% interval 0.882 to 0.913); mean set size 1.26; 81% of cases get a single label
- confidence tiers: high ≥ 0.8977406564656913, moderate ≥ 0.5, abstain below 0.5 or when the conformal set has more than 2 labels
- selective prediction: AURC 0.059 (95% CI 0.043 to 0.078); coverage at 5% error 0.72
The calibration split is drawn from the HAM10000 training release while the official test images were released separately, so a coverage gap between the two is a distribution-shift signal, not a bug.

## Robustness (corruption benchmark, 8 perturbations × 5 severities)
- balanced accuracy on 1000 test images: 0.685 clean; mean over perturbations 0.691 at severity 1 and 0.644 at severity 3
- three worst cells: gamma/5 = 0.368, jpeg/5 = 0.429, noise/5 = 0.434

## Subgroup audit (official test, metadata from the ISIC API)
- age_band, accuracy: worst subgroup **50-69** at 0.786 against 0.826 overall
- age_band, balanced accuracy: worst subgroup **50-69** at 0.647 against 0.680 overall
- age_band, melanoma sensitivity: worst subgroup **50-69** at 0.534 against 0.637 overall
- sex, accuracy: worst subgroup **male** at 0.811 against 0.826 overall
- sex, balanced accuracy: worst subgroup **male** at 0.642 against 0.680 overall
- sex, melanoma sensitivity: worst subgroup **female** at 0.564 against 0.637 overall
- site, accuracy: worst subgroup **Head and neck** at 0.738 against 0.826 overall (confidence interval entirely worse than overall)
- site, balanced accuracy: worst subgroup **Trunk** at 0.601 against 0.680 overall
- site, melanoma sensitivity: worst subgroup **Lower extremity** at 0.393 against 0.637 overall
Skin-tone labels do not exist in HAM10000, so performance by skin tone cannot be audited; the data skews to lighter skin and the model should be assumed weaker on darker skin.

## Trust signals measured on this model
- `unstable`: error 0.614 when raised (n=101) against 0.129 when not; difference +0.485 (CI +0.390 to +0.581). **flagged cases are wrong more often (difference CI excludes 0): the signal is usable**
- `low_quality`: error 0.145 when raised (n=433) against 0.203 when not; difference -0.057 (CI -0.102 to -0.010). **flagged cases are wrong LESS often: the signal is miscalibrated**
- `abstain`: error 0.622 when raised (n=74) against 0.143 when not; difference +0.479 (CI +0.356 to +0.596). **flagged cases are wrong more often (difference CI excludes 0): the signal is usable**
- `ood_energy`: error 0.529 when raised (n=51) against 0.159 when not; difference +0.370 (CI +0.231 to +0.512). **flagged cases are wrong more often (difference CI excludes 0): the signal is usable**

## Known failure modes
- Misses melanoma at top-1 about 36% of the time; the calibrated confidence and abstention exist to catch part of this, and the report must say so.
- Weak classes: actinic keratosis / intraepithelial carcinoma (recall 0.42) and vascular lesions (0.60).
- Degrades under noise, strong gamma changes and heavy JPEG compression (see robustness).
- Energy-score out-of-distribution detection separates other modalities with AUROC 0.86 to 0.98, below the 0.95 target; the router and MedSigLIP-based detector (P1) are the intended guard.

## Contamination status
Clean on the official ISIC test set. MedSAM and MedGemma, used alongside this model, have unverified training overlap.
