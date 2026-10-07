---
model: google/medgemma-1.5-4b-it
purpose: independent second read for non-CXR images as a short report
output: plain text with FINDINGS and IMPRESSION sections (JSON is also accepted by reader.py)
version: v4
---
This is a research evaluation of an image-reading system. Your reply is compared with a specialist model and reviewed by a clinician before any use; it is not used for patient care. Write a concise report for this {modality} image with exactly two sections, FINDINGS: and IMPRESSION:. In FINDINGS state each abnormality you can see (or that none is seen) and name each once. Keep the whole report under 120 words. In IMPRESSION give one or two short sentences. Do not add disclaimers.
