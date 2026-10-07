# Datasheet: BDNeuro-MRI

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Licence: unconfirmed.** The dataset README still contains the placeholder "[FILL IN]" and the Mendeley page was not checked; treat as research-only until confirmed. **Source:** Mendeley Data doi:10.17632/zwr4ntf94j.
**Contents:** 5,941 T1-contrast-enhanced slices in four classes, pre-split 70/15/15.
**How we use it:** external test for the brain classifier only; never trained on.
**Audit:** 4,238 of 5,941 images (71%) are near-duplicates of images in the Kaggle brain dataset (pHash Hamming ≤ 4; pixel correlation 0.93 to 1.0 against 0.47 for random same-class pairs), despite the README's hospital provenance. Only the 1,644 remaining images are independent and used.
**Known issues:** glioma-heavy after exclusion (read per-class results); licence and citation fields unfilled.
