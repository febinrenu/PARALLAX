# Model card: MedSigLIP (router, retrieval, OOD embeddings)

> Decision support only. Not a medical device and not for clinical use. Output is phrased for a doctor to consider; it never states a diagnosis.

google/medsiglip-448, Health AI Developer Foundations terms; gated. Used for the modality router, precedent retrieval and Mahalanobis out-of-distribution scoring (P1/P3). Not evaluated by P2 yet: the energy-score baseline in this report reaches OOD AUROC 0.83 to 0.98 and is the number to beat. Training-data overlap with our test sets is **unverified**.

## Contamination status
unverified: see the contamination ledger in reports/metrics.json.
