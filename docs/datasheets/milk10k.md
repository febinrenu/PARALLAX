# Datasheet: MILK10k

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Licence:** CC BY-NC 4.0 (non-commercial). **Source:** ISIC Archive, doi:10.34970/648456 (MILK study team).
**Contents:** 5,240 lesions, each with a clinical close-up and a dermoscopic image (10,480 images), 11 diagnostic classes (7 shared with HAM10000 plus squamous cell carcinoma / keratoacanthoma, other benign, other malignant and inflammatory), age in 5-year bands, sex, anatomic site and a 6-level skin-tone grade (0 very dark to 5 very light, distinct from Fitzpatrick types). 95.7% of lesions have histopathology.
**How we use it:** external test of the skin classifier (dermoscopic images only) and the only skin-tone audit. Never trained on.
**Audit:** 96 images are near-duplicates of HAM10000 images (the metadata notes that some lesions were previously in ISIC) and are excluded; 576 images of classes the model does not cover are excluded (ben_oth 44, inf 50, mal_oth 9, sccka 473); 4,566 images remain.
**Known issues:** basal cell carcinoma is over half of the remaining images; skin-tone grade 0 has only 6 lesions (grades 0 and 1 are merged); a different clinical mix from HAM10000, so accuracy differences mix skin tone with dataset shift.
