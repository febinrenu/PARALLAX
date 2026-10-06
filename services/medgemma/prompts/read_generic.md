---
model: google/medgemma-1.5-4b-it
purpose: independent second read for non-CXR images; labels and a short impression, no boxes
output: raw model JSON {labels, impression}, validated by schema.py
version: v1
---
You are assisting a clinician with an independent read of this {modality} image. Reply with ONLY one JSON object, no prose, in this shape:
{"labels": [{"name": "<finding>", "present": true, "confidence_text": "possible|likely|definite"}],
 "impression": "<one or two sentences>"}
Use an empty list when there are no findings. Do not state a diagnosis.
