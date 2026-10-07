# Model card: MedGemma 1.5 4B (second reader)

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

google/medgemma-1.5-4b-it, Health AI Developer Foundations terms. Zero-shot generalist reader and second opinion; run 4-bit (nf4) on a local GPU or Kaggle. Owner of its evaluation: P3 (reports/concordance.json). Measured on FracAtlas (275-image prefix of the evaluation batch): sensitivity about 0.19 at 99.5% specificity, so it is an audit flag for bone, not a reason to downgrade a finding. Training-data overlap with our test sets is **unverified**.

## Contamination status
unverified: see the contamination ledger in reports/metrics.json.
