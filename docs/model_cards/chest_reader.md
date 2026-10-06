# Model card: Chest X-ray reader

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Task:** multi-label chest radiograph reading (18 findings) with Grad-CAM localisation and anatomical zone naming.
**Models:** TorchXRayVision DenseNet121 weight sets, Apache-2.0, **not trained by this project**: `chex` (CheXpert), `mimic_ch` (MIMIC-CXR with CheXpert labels) and `all` (NIH, PadChest, CheXpert, MIMIC-CXR, Google, OpenI and RSNA). The product demo uses `all` for best accuracy; validation uses the external sets.

## Intended use and out-of-scope use
A second opinion on a frontal chest radiograph. Not for lateral views, paediatric films, or as a stand-alone reader.

## Evaluation: RSNA Pneumonia Detection Challenge, patient-level split (test n = 17,785; calibration n = 7,621)
Target: RSNA class Lung Opacity versus the other two classes (the challenge's pneumonia-positive). Prevalence 0.232.

| Weights | Trained on RSNA? | AUROC, Lung Opacity (95% CI) | AUROC, Pneumonia output | Status |
|---|---|---|---|---|
| chex | no | 0.785 (95% CI 0.777 to 0.793) | 0.762 | external, clean |
| mimic_ch | no | 0.749 (95% CI 0.741 to 0.757) | 0.701 | external, clean |
| all | **yes** | 0.875 (95% CI 0.869 to 0.881) | 0.836 | **contaminated**, reference only |

The contaminated weights look about 9 AUROC points better than the cleanest external set; the external numbers are the honest ones.

## Calibration and abstention (chex weights)
The outputs are operating-point-scaled probabilities, not logits, so 0.5 is each label's operating point, not a calibrated 50%. Platt scaling on logit(score), fitted on half of the RSNA calibration split:
- ECE 0.220 (95% CI 0.213 to 0.227) → 0.018 (95% CI 0.014 to 0.023)
- false-negative-rate control for Lung Opacity, target 10%: miss rate 0.103 (exact interval 0.094 to 0.112) with false-positive rate 0.482
- abstention band [0.4, 0.6]: 16% of images; accuracy outside the band 0.823 against 0.534 inside
Only Lung Opacity and Pneumonia can be calibrated against RSNA; the other 16 labels have no ground truth here and are reported as uncalibrated.

## Localisation (Grad-CAM++ against RSNA boxes, 1,000 test positives with boxes)
| Weights | Pointing game (95% CI) | Mean best IoU | IoU ≥ 0.3 |
|---|---|---|---|
| chex | 0.299 (95% CI 0.271 to 0.326) | 0.154 | 0.12 |
| mimic_ch | 0.196 (95% CI 0.171 to 0.220) | 0.100 | 0.03 |
| all (contaminated) | 0.639 (95% CI 0.609 to 0.669) | 0.231 | 0.30 |
Chance for a random point is 0.11 and for the image centre 0.10. With external weights the heatmap hits the annotated region only about a third of the time, so a heatmap is not evidence on its own: the faithfulness test (P1) decides whether it counts.

## Robustness (chex weights)
- AUROC on 500 test images: 0.816 clean; mean over perturbations 0.816 at severity 1 and 0.806 at severity 3
- three worst cells: contrast/5 = 0.539, contrast/4 = 0.708, gamma/5 = 0.718

## Subgroup audit (chex weights)
- age_band, auroc: worst subgroup **70+** at 0.733 against 0.785 overall (confidence interval entirely worse than overall)
- age_band, false negative rate at control threshold: worst subgroup **<30** at 0.163 against 0.103 overall (confidence interval entirely worse than overall)
- sex, auroc: worst subgroup **M** at 0.778 against 0.785 overall
- sex, false negative rate at control threshold: worst subgroup **M** at 0.110 against 0.103 overall
- view, auroc: worst subgroup **AP** at 0.699 against 0.785 overall (confidence interval entirely worse than overall)
- view, false negative rate at control threshold: worst subgroup **PA** at 0.250 against 0.103 overall (confidence interval entirely worse than overall)

## Trust signals measured on this model
- `unstable`: too few flagged or unflagged cases to judge (n flagged 8)
- `low_quality`: error 0.226 when raised (n=226) against 0.215 when not; difference +0.010 (CI -0.059 to +0.082). **no reliable difference in error rate between flagged and unflagged cases: do not use this signal to downgrade findings**
- `abstain`: error 0.524 when raised (n=84) against 0.159 when not; difference +0.365 (CI +0.250 to +0.475). **flagged cases are wrong more often (difference CI excludes 0): the signal is usable**

## Known failure modes
- Lower accuracy on portable (AP) films and on patients over 70.
- RSNA's "Lung Opacity" is a broader label than pneumonia; a positive is a reason to look, not a diagnosis.
- The reader sees only the central square of non-square images.

## Contamination status
chex and mimic_ch: external and clean. all: trained on RSNA (and on NIH ChestX-ray14, the source of RSNA's images): contaminated on this test set and labelled as such everywhere it is shown.
