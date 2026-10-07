# Precedent indices (P3)

The precedents stage returns the five nearest confirmed training cases for each finding. It needs one index per modality and, where the
licence allows, small thumbnails. The data is **not in git** (embeddings of licensed images, and about 130 MB). P3 sends it as
`retrieval_pack.zip`.

## Install

Unzip into `ml/artifacts/retrieval/` (the zip's paths are relative to that folder), or unzip anywhere and set `MEDPROOF_RETRIEVAL_DIR` to it.

```
ml/artifacts/retrieval/
  ham10000_medsiglip-448.{json,npy}     skin       6,803 images
  fracatlas_medsiglip-448.{json,npy}    bone       2,447 images
  brain_mri_medsiglip-448.{json,npy}    brain      3,349 images
  rsna_medsiglip-448.{json,npy}         chest      7,621 images
  thumbs/fracatlas/<case id>.png        2,447 files
  thumbs/brain_mri/<case id>.png        3,349 files
```

The embedder is `google/medsiglip-448` (gated: accept the terms on Hugging Face and set `HF_TOKEN`). The precedents stage uses only its **image tower**
through `AutoImageProcessor` and `AutoModel`, so it does not need the tokenizer files the automatic router needs. The first study in a process loads the
model (10 to 45 s on a CPU), later ones reuse it, so warm it at server start.

## What a precedent shows

| Dataset | Licence | Thumbnail | What the doctor sees |
|---|---|---|---|
| FracAtlas (bone) | CC BY 4.0 | yes (credit FracAtlas) | picture, true label, similarity |
| Brain MRI (Kaggle) | CC0 as stated on the dataset page (it merges Br35H, SARTAJ and Figshare, which have their own terms) | yes | picture, true label, similarity |
| HAM10000 / ISIC 2018 (skin) | CC BY-NC 4.0 | **no** | true label and similarity only |
| RSNA (chest) | competition rules, not redistributable | **no** | true label and similarity only |

This is enforced in code (`THUMBNAIL_DATASETS` in `backend/medproof/reasoning_stages.py`): even if a thumbnail file exists for a non-cleared dataset it is
never copied into a study, and its `thumb_ref` is the empty string. For a cleared dataset the stage copies the PNG into the study's artifact folder and sets
`thumb_ref` to `/studies/{study_id}/artifacts/precedent_{dataset}_{case_id}.png`, which the API already serves. A missing file also gives an empty `thumb_ref`.

## How good the precedents are (precision at 5, 95% interval, against picking five at random)

| Index | Precision@5 | Chance |
|---|---|---|
| Bone (FracAtlas) | 0.84 (0.82 to 0.87) | 0.70 |
| Skin (HAM10000) | 0.67 (0.64 to 0.69) | 0.47 |
| Brain MRI | 0.92 (0.90 to 0.94) | 0.27 |
| Chest (RSNA, reference set is the calibration split) | 0.67 (0.65 to 0.69) | 0.36 |

The weak spot is the rare class in each (fracture 0.48, melanoma 0.43, chest opacity 0.57), so precedents for a rare finding are less reliable than for a common one.
Full numbers: `reports/retrieval_<dataset>_medsiglip-448.json`.

## Rebuild

```
python -m ml.eval_p3.build_retrieval --dataset <fracatlas|ham10000|brain_mri|rsna> --embedder medsiglip     # index and report
python -m ml.eval_p3.build_retrieval --dataset <fracatlas|brain_mri> --thumbs-only                           # thumbnails only
```

An index is built from the training split only (the calibration split for chest, which has no training split), and each query leaves out its own image and its own patient or lesion group.
