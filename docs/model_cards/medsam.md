# Model card: MedSAM (box-prompted segmentation)

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

flaviagiammarino/medsam-vit-base, Apache-2.0. Gives a lesion mask from a box for skin images. Zero-shot evaluation on ISIC 2018 Task 1 (Jaccard) belongs to P1.12 and has not run. Its training corpus included public skin-lesion data; the overlap with ISIC 2018 Task 1 is **unverified**.

## Contamination status
unverified: see the contamination ledger in reports/metrics.json.
