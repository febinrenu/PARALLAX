# progress.md — Parallax live state

> Single source of truth for what is happening right now. Every session reads this after CLAUDE.md and plan.md.
> **Rules:** edit only your own status rows and your own role section. The shared log at the bottom is append-only: add new entries at the very end, never edit old ones. Format is in plan.md §12.3. `.gitattributes` sets `merge=union` for this file.

## Team

| Role | Person | Machine / GPU | Groq key variable |
|---|---|---|---|
| P1 Imaging Core | | | GROQ_KEY_AUDIO |
| P2 Data, Training & Validation | | laptop, RTX 4050 6 GB | GROQ_KEY_JUDGE |
| P3 Clinical Reasoning & Trust | | | GROQ_KEY_REPORT |
| P4 Experience & Platform | | | GROQ_KEY_EXTRACT |

## Setup (plan.md §6)

- [ ] Everyone: repo access, git identity set, Claude Code on Sonnet 5.5
- [ ] Everyone: own Groq key in local `.env`
- [ ] Everyone: `git config core.hooksPath .githooks` (hook verified to reject assistant mentions)
- [ ] P2: Kaggle token; RSNA rules accepted; BDNeuro-MRI downloaded
- [ ] P3: HF token; MedGemma 1.5 + MedSigLIP terms accepted; Kaggle phone-verified
- [ ] P4: skeleton committed; `make setup` green on a fresh clone
- [ ] Groq `/models` checked at T+0; IDs in config match (record below)
- [ ] Local NVIDIA GPU available? (who, VRAM):

## Gates

| Gate | Target time | Criterion | Status | Notes |
|---|---|---|---|---|
| G0 | T+1 | Contract frozen, fixtures committed | todo | |
| G1 | T+4 | CXR end to end with real output | todo | |
| G2 | T+7 | All 4 modalities end to end; status rule + firewall live | todo | |
| G3 | T+9 | Metrics frozen in `reports/` | todo | |
| G4 | T+11 | Feature freeze | todo | |
| Submit | T+13 | Repo link submitted via form; backup video recorded | todo | |

## Status board

States: `todo` · `doing` · `blocked` · `done` · `cut`


### P1 — Imaging Core

| WP | Task | State | Branch | Updated | Note |
|---|---|---|---|---|---|
| P1.1 | Decode PNG/JPG/DICOM + hashing | done | p1/intake | 2026-10-06 | `intake/decode.py`, `intake/preprocess.py` |
| P1.2 | DICOM PHI scrub | done | p1/intake | 2026-10-06 | `intake/phi.py` |
| P1.3 | Quality gate | done | p1/intake | 2026-10-06 | thresholds provisional, refit on real data |
| P1.4 | Router (MedSigLIP probe) | todo | | | |
| P1.5 | OOD score | todo | | | |
| P1.6 | CXR reader + anatomy zones | todo | | | |
| P1.7 | Bone reader wrapper | todo | | | |
| P1.8 | Brain + skin inference wrappers | todo | | | |
| P1.9 | Faithfulness test (D1) | todo | | | |
| P1.10 | Saliency sanity check (D2) | todo | | | |
| P1.11 | Stability score (D5) | doing | p1/intake | 2026-10-06 | perturbation library done; flip-rate logic todo |
| P1.12 | MedSAM service | todo | | | |

### P2 — Data, Training & Validation

| WP | Task | State | Branch | Updated | Note |
|---|---|---|---|---|---|
| P2.1 | Dataset downloads + manifests | doing | p2/data | 2026-10-06 | credential-free sets done; Kaggle sets wait for token |
| P2.2 | Leakage audit + group splits | doing | p2/data | 2026-10-06 | fracatlas + ham10000 done; brain, lgg, rsna wait for data |
| P2.3 | Brain classifier (Kaggle) | doing | p2/data | 2026-10-06 | notebook being written |
| P2.4 | Brain segmenter (Kaggle) | doing | p2/data | 2026-10-06 | notebook being written |
| P2.5 | Skin classifier (Kaggle) | doing | p2/data | 2026-10-06 | notebook being written |
| P2.6 | Bone detector (Kaggle) | doing | p2/data | 2026-10-06 | notebook being written |
| P2.7 | CXR calibration + external validation | todo | | | |
| P2.8 | Temperature scaling + conformal | todo | | | |
| P2.9 | Selective prediction | todo | | | |
| P2.10 | Trust-signal validation (D13) | todo | | | |
| P2.11 | Subgroup audit | todo | | | |
| P2.12 | Corruption benchmark | todo | | | |
| P2.13 | Model cards + datasheets | todo | | | |
| P2.14 | make eval | todo | | | |

### P3 — Clinical Reasoning & Trust

| WP | Task | State | Branch | Updated | Note |
|---|---|---|---|---|---|
| P3.1 | MedGemma service + batch reads | todo | | | |
| P3.2 | Generalist reader + concordance | todo | | | |
| P3.3 | Groq pool | todo | | | |
| P3.4 | Synthetic note generator | todo | | | |
| P3.5 | Fact extraction with spans | todo | | | |
| P3.6 | Injection guard | todo | | | |
| P3.7 | Contradiction rules | todo | | | |
| P3.8 | Retrieval / precedents | todo | | | |
| P3.9 | Slot-filled report | todo | | | |
| P3.10 | Hallucination firewall | todo | | | |
| P3.11 | Entailment judge | todo | | | |
| P3.12 | Voice (Whisper) | todo | | | |
| P3.13 | FHIR export | todo | | | |

### P4 — Experience & Platform

| WP | Task | State | Branch | Updated | Note |
|---|---|---|---|---|---|
| P4.1 | Repo skeleton + Makefile + tooling | todo | | | |
| P4.2 | Contract + fixtures (G0) | todo | | | |
| P4.3 | Pipeline orchestrator | todo | | | |
| P4.4 | API + SSE | todo | | | |
| P4.5 | Evidence ledger | todo | | | |
| P4.6 | Web app pages | todo | | | |
| P4.7 | WebGL viewer | todo | | | |
| P4.8 | Docker + CI + e2e + a11y | todo | | | |
| P4.9 | Offline demo pack | todo | | | |
| P4.10 | README | todo | | | |


## Role sections

### P1 — Imaging Core
**Now:** P1.1 to P1.3 done and tested (121 tests passing). Perturbation library pulled forward from P1.11.
**Next:** P1.6 CXR reader (unblocks P2.7), then P1.4 router, P1.9 faithfulness, P1.11 flip-rate logic.
**Blockers:** none.
**For P2 (available now on branch `p1/intake`):**
- Decode any upload: `from medproof.intake.decode import load_image` returns `DecodedImage` (`.display` uint8, `.analysis` float32 in [0,1], `.sha256`). DICOM handles rescale slope/intercept, window tags and MONOCHROME1 (RSNA). Raises `DecodeError` on bad input. Use this for pHash and for every eval loader so you see the pixels the product sees.
- Training and serving preprocessing: `from medproof.intake.preprocess import prepare, get_spec`; specs `cxr_xrv`, `brain_effnet`, `skin_cls`, `bone_yolo`. Call `prepare(decoded_or_float_array, "brain_effnet")` to get a float32 CHW tensor. Change a spec only via a Decisions entry so train and serve stay equal.
- Corruption benchmark: `from medproof.verify.perturbations import apply, NAMES, SEVERITIES`; `apply(name, img_float01, severity 1..5, seed)`. Eight perturbations, deterministic for a given seed.
- Quality gate: `from medproof.intake.quality import assess, QualityConfig`; `assess(decoded, "cxr")` gives `.passed`, `.reasons` (code, level, fix), `.metrics`. Thresholds are in `QualityConfig` and were set on a synthetic phantom; refit them on real validation images (target: at most 5% of clean images flagged).
- Reader output shape: `medproof.readers.base.ReaderOutput` (stub, final with P1.6).
- Setup: `cd backend`, Python 3.12 venv, `pip install numpy pillow opencv-python-headless pydicom scipy pydantic pytest hypothesis`, then `python -m pytest -q`.

### P2 — Data, Training & Validation
**Now:** P2.1 and P2.2 are done for everything that needs no credentials. Downloaded and hashed: ISIC 2018 Task 3 train and official test, HAM10000 metadata, ISIC API metadata for the official test (age, sex, site, lesion_id), FracAtlas. Splits and `reports/leakage.json` exist for `fracatlas` and `ham10000`. Branch `p2/data`. Building the four training notebooks (P2.3 to P2.6).
**Next:** four notebooks under `ml/train/notebooks/`, then brain_mri, lgg_seg and rsna splits as soon as their data arrives, then P2.7 onward.
**Blockers:** (1) Kaggle token in `.env` (brain MRI, LGG, RSNA need it; no `.env` exists on this machine). (2) RSNA competition rules not accepted yet. (3) BDNeuro-MRI manual download. (4) This machine's C: drive is full (27 MB free): pip and pytest temp must go to D:.
**Machine:** laptop, NVIDIA RTX 4050 6 GB (can run the smaller training jobs if Kaggle GPU quota runs out).
**For P3 (all on branch `p2/data`, nothing here needs a P2 reply):**
- Run `python ml/data/download.py core` once on your machine, or point `PARALLAX_DATA_ROOT` at a folder that has `<dataset>/...` as laid out in `ml/data/splits/*.csv` (`source_dir` + `relpath`). On Kaggle, symlink attached inputs into that layout.
- `from ml.data.common import load_split, image_path, load_eval_index`. `load_split("ham10000")` gives one row per image with `label`, `split` (`train/val/cal/test`, `official_test` for the ISIC test set, `dropped` = removed by the audit), `age`, `sex`, `site_general` (note seeds), `group`, `eval_batch`.
- `ml/data/eval_index.json` lists, per dataset, the test split and a bounded batch subset (`eval_batch == True`, at most 1,000 images, label-stratified, seeded). Use it for the MedGemma batch reads (P3.1/3.2): fracatlas 569 test images, ham10000 official test 1,000 of 1,512.
- Retrieval index (P3.8): build it from `split == "train"` only, so a precedent can never be a test image.
- Notes seeds: FracAtlas rows carry `body_part`, `view`, `hardware`, `label`; HAM rows carry `age`, `sex`, `site_general`, `label`. `gen_notes.py` is yours; P2 does not touch it.
- brain_mri, lgg_seg and rsna split files appear after the Kaggle token arrives; `eval_index.json` updates itself.
**Kaggle jobs:** (notebook slug, started, status, artifact hash)
- none launched yet

### P3 — Clinical Reasoning & Trust
**Now:**
**Next:**
**Blockers:**
**MedGemma endpoint:** (where it runs, current URL, last health check)

### P4 — Experience & Platform
**Now:**
**Next:**
**Blockers:**

## Decisions

Deviations from plan.md, newest last. Format: `date · role · decision · evidence`.

- 2026-10-06 · P1 · Pulled `verify/perturbations.py` forward from P1.11 · shared by the quality-gate fixtures, the stability score and P2.12, so it is defined once.
- 2026-10-06 · P1 · Added `intake/preprocess.py` (not in plan) · one preprocessing path for training notebooks and inference prevents train/serve skew.
- 2026-10-06 · P1 · Blur metric is Laplacian variance divided by intensity variance, computed on a 512 px copy · raw Laplacian variance changes with contrast and resolution; test shows scores within 35% across a 512 to 1024 px resize.
- 2026-10-06 · P1 · CXR tilt is estimated from the rotation that maximises left-right mirror symmetry (with an off-centre shift allowed), not the lung-mask axis · the anatomy segmenter arrives with P1.6; replace then. Only a tilt with symmetry gain of at least 0.03 is reported.
- 2026-10-06 · P1 · PHI scrub keeps PatientAge and PatientSex, blanks names and identifiers, removes free-text descriptions, shifts dates by one random per-study offset that is not recorded, remaps UIDs, drops private tags, and warns on burned-in annotation · the context checks need age and sex; this is a PS3.15 subset, not a certified de-identifier.
- 2026-10-06 · P1 · Quality thresholds are provisional values tuned on a synthetic phantom: the clean phantom passes with zero reasons across 8 seeds, and 100% of noise, blur and contrast degradations at severity 4 and 5 fail the gate · must be refit on real validation images.
- 2026-10-06 · P1 · Python 3.12 venv with pip instead of 3.11 with uv (uv not installed here); `backend/pyproject.toml` is minimal for the platform owner to absorb · no `make smoke` exists yet, so verification is `python -m pytest` in `backend/`.
- 2026-10-06 · P1 · `core/schemas.py` is a verbatim copy of the section 4 sketch because the contract was not yet committed · platform owner replaces it; P1 code only uses `StageResult`.
- 2026-10-06 · P2 · FracAtlas source corrected: figshare file 65518038 (v7, CC BY 4.0, md5 fe9da2c7...; dataset.csv lists 4,083 images, 719 fractured, the figshare text says 4,073), resolved at run time through the figshare API · plan.md's file id 43283628 returns HTTP 202 with no body.
- 2026-10-06 · P2 · HAM10000 metadata is fetched from Harvard Dataverse (doi:10.7910/DVN/DBW86T, no login); Kaggle is the fallback only · removes the Kaggle-token dependency for metadata.
- 2026-10-06 · P2 · Official ISIC 2018 Task 3 test metadata (age, sex, site, lesion_id) is fetched from the ISIC API · lets the official test set carry lesion-grouped CIs and a subgroup audit.
- 2026-10-06 · P2 · Splits get a separate conformal calibration set: train 70 / val 10 / cal 10 / internal test 10 (HAM10000, brain), 65/10/10/15 (FracAtlas), LGG fold 0 of a 5-fold patient CV · temperature scaling uses val, conformal uses cal, so neither sees test.
- 2026-10-06 · P2 · Own 64-bit DCT pHash in `ml/data/phash.py` instead of `imagehash`; duplicate groups = native id (lesion, patient) plus Hamming <= 4 edges; official test sets are never modified · avoids a PyWavelets build on new Pythons and guarantees a duplicate cannot straddle splits.
- 2026-10-06 · P2 · Brain classifier is trained twice (original Kaggle split vs leakage-free split) · measures the inflation gap instead of estimating it.
- 2026-10-06 · P2 · Bootstrap: percentile cluster bootstrap, B=2000, class-stratified for balanced accuracy, BCa for per-patient Dice; bone sensitivity at 90% specificity uses a threshold fixed on val · plan said B=1000.
- 2026-10-06 · P2 · YOLO26n (released 2026-01-14) for the bone detector, yolo11n as fallback, via config · closes the [VERIFY] on the latest nano model.
- 2026-10-06 · P2 · `.gitignore` carve-out `!ml/data/` plus `ml/data/raw/` ignored (the `data/` rule hid `ml/data/`); `ml/Makefile.inc` supplied for P4 to include · P4 owns both files, so P4 please ack.
- 2026-10-06 · P2 · `gen_notes.py` (synthetic notes) is left to P3; P2 supplies seed columns (age, sex, site, body part, label) in `ml/data/splits/*.csv` · P3.4 owns note content and injection cases.
- 2026-10-06 · P2 · Flag for P1/P4: the LGG segmenter is trained on 3-channel TCGA data; the brain classifier data is single-channel T1-CE. Masks on classifier-style uploads are out of domain, and the model card will say so. Channel-replication augmentation reduces the gap but does not remove it.

## Contract change requests

Format: `id · proposer · change · affected roles · P4 ack (yes/no) · applied in commit`.

## Verified facts log

Record anything marked [VERIFY] in plan.md once checked. Format: `date · role · item · result · source`.

## Shared log (append-only, newest at the bottom)

<!-- Add entries below this line using the format in plan.md §12.3. Never edit earlier entries. -->

### 2026-10-06 · P1 · WP P1.1, P1.2, P1.3
Did: DICOM/PNG/JPG decode with rescale, windowing and MONOCHROME1 inversion; shared preprocessing specs; eight seeded perturbations; PHI scrub with receipt; quality gate with fix text per failure; intake stage adapter returning StageResult.
State: P1.1 to P1.3 done. Perturbation library done early (P1.11 logic still todo).
Verified: `cd backend && python -m pytest -q` gives 121 passed. Includes a pydicom-built MONOCHROME1 fixture, a PHI sentinel test on the serialized bytes, and six degraded images each giving the right reason.
Next: CXR reader and anatomy zones (P1.6) so P2.7 can calibrate; then router.
Decisions: see Decisions, entries dated 2026-10-06 for P1.
Contract change requests: none.
