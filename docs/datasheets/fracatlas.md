# Datasheet: FracAtlas

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

**Licence:** CC BY 4.0. **Source:** figshare doi:10.6084/m9.figshare.22363012 (v7).
**Contents:** 4,083 radiographs (hand, leg, hip, shoulder, mixed), 719 fractured, with COCO, YOLO, VOC and VGG annotations and body-part, view and implant metadata. Collected at three hospitals in Bangladesh.
**How we use it:** bone detector training and evaluation (65/10/10/15 split, stratified by body part and fracture status, negatives kept).
**Audit:** 341 near-duplicate pairs; 281 images removed; 2 images sit in the "Non_fractured" folder although the metadata and boxes mark them fractured (we follow the metadata and boxes); 59 negative images are truncated by a few bytes at the source and are flagged; the publisher's own split files cover only fractured images and were not used.
**Known issues:** no patient identifiers (other views of a patient may cross splits); single-country source; fractures are small and annotated with loose boxes; the stated image count differs between the paper (4,083) and the figshare text (4,073).
