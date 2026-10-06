# Model card: Brain tumour segmenter

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Registry:** `brain_seg@d351f062` · weights sha256 `90eba65ab2b7c2b6…` · split `8637d82d7d71…` · commit `d351f062`
**Task:** tumour segmentation on brain MRI slices.
**Architecture:** U-Net with a ResNet34 encoder (segmentation_models_pytorch), 256 px, Dice + BCE loss, 20 epochs, fold 0 of a patient-level 10-fold split (shipped model).
**Licence:** MIT code; trained on LGG MRI Segmentation (CC BY-NC-SA 4.0): non-commercial, share-alike.

## Intended use and out-of-scope use
Outlining the extent of an already-detected tumour on a slice. It is trained on lower-grade glioma only; it must not be used to decide whether a tumour exists or on other tumour types.

## Training data
LGG MRI Segmentation (TCGA, 110 patients, 3,929 slices, 3 channels: pre-contrast, FLAIR, post-contrast). Patients are never split across sets; two patients are merged into one group only when tumour-bearing slices are near-duplicates (28 patients merged). Channel-replication augmentation lets the network accept one-channel input, but this does not remove the domain gap to single-channel T1-CE uploads.

## Evaluation (held-out patient groups, n = 9)
- **Mean per-patient Dice 0.831 (95% CI 0.716 to 0.890)** (BCa bootstrap over patients); pooled Dice 0.882
- per-patient Dice ranges 0.50 to 0.93
- mean HD95 9.3 (95% CI 6.5 to 15.1) pixels (the dataset gives no pixel spacing)
- slice-level tumour detection AUROC 0.976
Nine test groups is a small sample, so the interval is wide.

## Robustness (photometric perturbations only; a rotation or crop would also move the ground-truth mask)
- pooled Dice on 372 test images: 0.882 clean; mean over perturbations 0.875 at severity 1 and 0.834 at severity 3
- three worst cells: noise/5 = 0.275, noise/4 = 0.470, blur/5 = 0.629

## Known failure modes
- Out of domain for meningioma, pituitary tumours and single-channel T1-CE slices (the classifier's data). Masks on those inputs are unvalidated.
- Weakest patients have few tumour slices; Dice is lowest when the tumour is small.
- Heavy noise reduces Dice sharply.

## Contamination status
Clean: patient-level split, trained and tested on TCGA-LGG only. The reference model `mateuszbuda/brain-segmentation-pytorch` was trained on the same data and is never reported as our result.
