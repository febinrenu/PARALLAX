# Datasheet: Brain Tumor MRI Dataset (Kaggle, masoudnickparvar)

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Licence:** CC0 (as stated on the Kaggle page; the sources have their own terms). **Source:** a merge of Br35H, SARTAJ and Figshare (Cheng et al.) brain tumour MRI sets.
**Contents:** 7,200 2D slices in four classes (glioma, meningioma, no tumour, pituitary), original Training and Testing folders.
**Collection:** public research datasets of unclear patient-level provenance; no patient identifiers.
**How we use it:** brain classifier training and evaluation after the audit.
**Leakage audit:** 5,856 near-duplicate pairs; 2,019 of them cross the original Training and Testing folders; 2,375 images removed as duplicates or label noise (7 for conflicting labels). Slices within Hamming distance 6 are grouped so they stay in one split.
**Known issues:** merged sources with different acquisition styles; duplicate and near-duplicate slices; no patient identifiers, so the leakage-free split is an approximation; class balance shifts after deduplication (no-tumour loses the most images).
