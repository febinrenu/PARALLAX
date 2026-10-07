# Model card: Brain MRI classifier

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Registry:** `brain_cls@d351f062` · weights sha256 `184810f022787f8e…` · split `ae05442fab47…` · commit `d351f062`
**Task:** four-class brain MRI slice classification (glioma, meningioma, no tumour, pituitary).
**Architecture:** EfficientNet-B0 (timm `efficientnet_b0.ra_in1k`), 224 px, 10 epochs, AdamW 3e-4, label smoothing 0.1.
**Licence:** Apache-2.0 backbone; trained on a CC0 dataset.

## Intended use and out-of-scope use
A second opinion on a single 2D slice. Not for CT, not for other sequences or planes the data lacks, not for follow-up or volumetric reading.

## Training data
Brain Tumor MRI Dataset (Kaggle; a merge of Br35H, SARTAJ and Figshare). It has no patient identifiers. The audit removed 2,375 of 7,200 images as duplicates or label noise and grouped similar slices (pHash Hamming ≤ 6) so near-identical slices stay in one split. Split 70/10/10/10 on the cleaned set.

## Evaluation
- **Leakage-free test split (563 images): accuracy 0.940 (95% CI 0.921 to 0.958)**, balanced accuracy 0.936 (95% CI 0.911 to 0.958), macro-F1 0.936 (95% CI 0.914 to 0.956); per-class recall glioma 0.95, meningioma 0.91, notumor 0.91, pituitary 0.98
- **External (BDNeuro-MRI, 1644 images with no duplicate in the training source): accuracy 0.755 (95% CI 0.735 to 0.775)**; balanced accuracy 0.740 (95% CI 0.718 to 0.762); per-class recall glioma 0.79, meningioma 0.90, notumor 0.83, pituitary 0.43. Accuracy falls about 18 points outside the training sources; pituitary cases are the weakest.

## Benchmark inflation (the leakage audit)
A diagnostic model trained on the original Kaggle split scores 0.942 (95% CI 0.928 to 0.954) on the original Testing folder, 0.903 (95% CI 0.878 to 0.925) after removing the 727 images that have a near-duplicate in Training, and 0.865 (95% CI 0.835 to 0.894) after also removing 1051 images that are near neighbours of Training images (same scan, adjacent slice). The comparison is within one model, which isolates the leak.

## Calibration and abstention
- temperature T = 0.689 (fitted on the validation split); ECE 0.079 → 0.019 on the test split
- split-conformal (randomised APS, fitted on the calibration split), target coverage 90%: empirical coverage 0.940 (exact 95% interval 0.917 to 0.958); mean set size 1.00; 100% of cases get a single label
- confidence tiers: high ≥ 0.6333719592353397, moderate ≥ 0.5, abstain below 0.5 or when the conformal set has more than 2 labels
- selective prediction: AURC 0.014 (95% CI 0.005 to 0.020); coverage at 5% error 0.97
The leakage-free test split is slightly easier than the calibration split, so coverage lands above the 90% target (a conservative error).

## Robustness
- balanced accuracy on 563 test images: 0.936 clean; mean over perturbations 0.938 at severity 1 and 0.878 at severity 3
- three worst cells: noise/5 = 0.229, noise/4 = 0.327, gamma/5 = 0.543

## Subgroup audit
The dataset carries no age, sex or scanner metadata, so only the per-class audit is possible:
- class, recall: worst subgroup **meningioma** at 0.905 against 0.940 overall

## Trust signals measured on this model
- `unstable`: error 0.870 when raised (n=23) against 0.026 when not; difference +0.844 (CI +0.701 to +0.974). **flagged cases are wrong more often (difference CI excludes 0): the signal is usable**
- `low_quality`: error 0.061 when raised (n=495) against 0.059 when not; difference +0.002 (CI -0.057 to +0.059). **no reliable difference in error rate between flagged and unflagged cases: do not use this signal to downgrade findings**
- `abstain`: too few flagged or unflagged cases to judge (n flagged 5)
- `ood_energy`: error 0.241 when raised (n=29) against 0.051 when not; difference +0.191 (CI +0.044 to +0.361). **flagged cases are wrong more often (difference CI excludes 0): the signal is usable**

## Known failure modes
- Does not generalise to a different hospital's data without a large accuracy drop (see external evaluation).
- Severe noise (severity 4 and 5) collapses accuracy; images with fp16 overflow can yield invalid outputs on unusual inputs, so inference falls back to float32 for those rows.
- Slice-level only: no patient identifiers existed, so grouping by image similarity approximates patient grouping and may still leave some same-patient slices across splits.

## Contamination status
Clean on the leakage-free test split. The original Kaggle Testing folder is contaminated (shown only to measure inflation). BDNeuro-MRI is independent only for the non-duplicate images.
