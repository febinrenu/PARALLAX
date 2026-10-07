# Datasheet: RSNA Pneumonia Detection Challenge

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Licence/terms:** Kaggle competition rules (research use); images derive from the NIH ChestX-ray14 collection. Not redistributed in this repository.
**Contents:** 26,684 frontal chest DICOMs (stage 2 training set) with bounding boxes for lung opacity and a three-class label (Normal, No Lung Opacity / Not Normal, Lung Opacity); PatientAge, PatientSex and ViewPosition are in the headers.
**How we use it:** calibration (30%) and external validation (70%) of pretrained chest readers, plus localisation scoring against its boxes. No training.
**Audit:** 1,851 near-duplicate pairs; 1,278 images removed. Patient ids are unique per image in this set, so grouping is by duplicate cluster.
**Contamination:** TorchXRayVision `all` weights were trained on it; `chex` and `mimic_ch` were not. NIH ChestX-ray14 is the source of RSNA's images and is part of the `all` training mix.
**Known issues:** "Lung Opacity" is not pneumonia; labels are from radiologist consensus on a single frontal view; mix of AP and PA films with different case mix.
