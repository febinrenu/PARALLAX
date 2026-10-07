# Datasheet: LGG MRI Segmentation (TCGA-LGG, Buda et al.)

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Licence:** CC BY-NC-SA 4.0 (non-commercial, share-alike). **Source:** Kaggle `mateuszbuda/lgg-mri-segmentation`, derived from The Cancer Imaging Archive.
**Contents:** 110 patients, 3,929 axial slices with 3 channels (pre-contrast, FLAIR, post-contrast) and manual FLAIR tumour masks.
**How we use it:** segmenter training with a patient-level 10-fold split (fold 0 shipped).
**Audit:** slices are never removed. Near-duplicate slices from different patients are mostly dark top and bottom slices and carry no information, so patients merge into one group only when tumour-bearing slices match (28 patients merged).
**Known issues:** lower-grade glioma only; one anatomical plane; masks drawn on FLAIR; a few patients contribute many slices.
