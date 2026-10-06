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
| P1.6 | CXR reader + anatomy zones | done | p1/intake | 2026-10-07 | 67 reader tests incl. 5 on real weights; localization quality not yet benchmarked (P2.7) |
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
| P2.2 | Leakage audit + group splits | doing | main | 2026-10-07 | fracatlas, ham10000, brain_mri, lgg_seg done; rsna waits for data |
| P2.3 | Brain classifier (Kaggle) | doing | main | 2026-10-07 | queued, waiting for a free Kaggle GPU slot |
| P2.4 | Brain segmenter (Kaggle) | doing | main | 2026-10-07 | queued, waiting for a free Kaggle GPU slot |
| P2.5 | Skin classifier (Kaggle) | doing | main | 2026-10-07 | running on Kaggle since 01:00 |
| P2.6 | Bone detector (Kaggle) | doing | main | 2026-10-07 | running on Kaggle since 01:00 |
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
| P4.1 | Repo skeleton + Makefile + tooling | done | main | 2026-10-07 | Makefile, root + backend pyproject tooling, pre-commit, minimal `web/` (Vite+React+TS) |
| P4.2 | Contract + fixtures (G0) | done | main | 2026-10-07 | schemas.py unchanged (additive only); `core/vocab.py`, `core/status_rule.py`; JSON Schema + TS export; 4 fixtures |
| P4.3 | Pipeline orchestrator | done | main | 2026-10-07 | `pipeline.py` + `core/{context,config,cache}.py`; only intake + CXR reader stages exist today, new ones append to `PIPELINE` |
| P4.4 | API + SSE | done | main | 2026-10-07 | `api/{app,studies,system}.py`; FastAPI + sse-starlette; in-memory `core/store.py` (not persisted) |
| P4.5 | Evidence ledger | done | main | 2026-10-07 | `core/ledger.py`; single global hash-chained JSONL; `GET /ledger/verify`; doctor feedback logged |
| P4.6 | Web app pages | todo | | | |
| P4.7 | WebGL viewer | todo | | | |
| P4.8 | Docker + CI + e2e + a11y | todo | | | |
| P4.9 | Offline demo pack | todo | | | |
| P4.10 | README | todo | | | |


## Role sections

### P1 — Imaging Core
**Now:** P1.1 to P1.3 and P1.6 done and tested (191 passed, 0 skipped: `cd backend && python -m pytest -q`). Perturbation library pulled forward from P1.11.
**Next:** P1.4 router, then P1.9 faithfulness (re-scores through `CxrReader.score`), P1.11 flip-rate logic, P1.5 OOD, P1.10 saliency sanity.
**Blockers:** none.
**For P2 (available now on branch `p1/intake`):**
- Decode any upload: `from medproof.intake.decode import load_image` returns `DecodedImage` (`.display` uint8, `.analysis` float32 in [0,1], `.sha256`). DICOM handles rescale slope/intercept, window tags and MONOCHROME1 (RSNA). Raises `DecodeError` on bad input. Use this for pHash and for every eval loader so you see the pixels the product sees.
- Training and serving preprocessing: `from medproof.intake.preprocess import prepare, get_spec`; specs `cxr_xrv`, `brain_effnet`, `skin_cls`, `bone_yolo`. Call `prepare(decoded_or_float_array, "brain_effnet")` to get a float32 CHW tensor. Change a spec only via a Decisions entry so train and serve stay equal.
- Corruption benchmark: `from medproof.verify.perturbations import apply, NAMES, SEVERITIES`; `apply(name, img_float01, severity 1..5, seed)`. Eight perturbations, deterministic for a given seed.
- Quality gate: `from medproof.intake.quality import assess, QualityConfig`; `assess(decoded, "cxr")` gives `.passed`, `.reasons` (code, level, fix), `.metrics`. Thresholds are in `QualityConfig` and were set on a synthetic phantom; refit them on real validation images (target: at most 5% of clean images flagged).
- Reader output shape: `medproof.readers.base.ReaderOutput` (final with P1.6).
- **P2.7 handoff (CXR reader, available on `p1/intake` and `main`):**
  - Needs: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu` and `pip install torchxrayvision`. Weights download on first use to `~/.torchxrayvision/` (DenseNet 28 MB, anatomy PSPNet 273 MB).
  - Probabilities only: `CxrReader.load(CxrConfig(weights="densenet121-res224-chex")).score(img)` returns an (18,) float32 array aligned to `reader.labels` (empty-string labels are unsupported by that weight set, ignore them). For `mimic_ch` or `chex` external validation on RSNA use those weights; `-all` saw RSNA (contaminated). `MEDPROOF_CXR_WEIGHTS` overrides the default.
  - These outputs are NOT raw logits: torchxrayvision applies sigmoid then per-label operating-point scaling, so 0.5 is each label's operating threshold. Calibrate on these values as given and say so in the model card. `ReaderOutput.output_kind` says "probability".
  - Full read with localization: `CxrReader.load(anatomy_fn=AnatomySegmenter.try_load()).predict(img)` then `evidence.to_findings(out, cfg, artifact_dir=...)` gives contract `Finding` and `ImageEvidence` (`prob_calibrated` equals `prob_raw`, flag `uncalibrated`, `conformal_set=[]`: calibration fills these). Stage entry point: `readers.cxr.run(ctx)`.
  - Localization eval: boxes are `bbox_xyxy` in original-image pixels (max edge exclusive); heatmaps are float32 [0,1] on the original grid. The model reads only the central square of non-square images; heatmap outside it is exactly 0 and the finding carries flag `partial_view`. RSNA images are square, so no effect there.
  - Preprocessing: `intake.preprocess.prepare(img, "cxr_xrv")` now centre-crops to a square then resizes to 224 (TorchXRayVision's own convention). The spec changed from stretch to centre-crop, so any notebook using it for CXR should re-run; brain, skin and bone specs are unchanged.
- Setup: `cd backend`, Python 3.12 venv, `pip install numpy pillow opencv-python-headless pydicom scipy pydantic pytest hypothesis`, then `python -m pytest -q`.

### P2 — Data, Training & Validation
**Now:** P2.1 and P2.2 are done for everything that needs no credentials. Downloaded and hashed: ISIC 2018 Task 3 train and official test, HAM10000 metadata, ISIC API metadata for the official test (age, sex, site, lesion_id), FracAtlas. Splits and `reports/leakage.json` exist for `fracatlas` and `ham10000`. Merged to `main`. The four training notebooks (P2.3 to P2.6) are written under `ml/train/notebooks/` and tested.
**Next:** push `p2/data` (the notebooks clone it), launch the four jobs once the Kaggle token exists, then brain_mri, lgg_seg and rsna splits as soon as their data arrives, then P2.7 onward.
**Blockers:** (1) none for RSNA any more: rules accepted, download running. (2) BDNeuro-MRI licence: its README still says `[FILL IN]`; check the Mendeley page and record it before we show results on it.
**Machine:** laptop, NVIDIA RTX 4050 6 GB (can run the smaller training jobs if Kaggle GPU quota runs out).
**For P3 (all on branch `p2/data`, nothing here needs a P2 reply):**
- Run `python ml/data/download.py core` once on your machine, or point `PARALLAX_DATA_ROOT` at a folder that has `<dataset>/...` as laid out in `ml/data/splits/*.csv` (`source_dir` + `relpath`). On Kaggle, symlink attached inputs into that layout.
- `from ml.data.common import load_split, image_path, load_eval_index`. `load_split("ham10000")` gives one row per image with `label`, `split` (`train/val/cal/test`, `official_test` for the ISIC test set, `dropped` = removed by the audit), `age`, `sex`, `site_general` (note seeds), `group`, `eval_batch`.
- `ml/data/eval_index.json` lists, per dataset, the test split and a bounded batch subset (`eval_batch == True`, at most 1,000 images, label-stratified, seeded). Use it for the MedGemma batch reads (P3.1/3.2): fracatlas 569 test images, ham10000 official test 1,000 of 1,512.
- Retrieval index (P3.8): build it from `split == "train"` only, so a precedent can never be a test image.
- Notes seeds: FracAtlas rows carry `body_part`, `view`, `hardware`, `label`; HAM rows carry `age`, `sex`, `site_general`, `label`. `gen_notes.py` is yours; P2 does not touch it.
- brain_mri, lgg_seg and rsna split files appear after the Kaggle token arrives; `eval_index.json` updates itself.
**Kaggle jobs:** (notebook slug, started, status, artifact hash)
- rishijayanath/parallax-p2-5-skin-lesion-classifier: started 2026-10-07 01:00, running (15 epochs, ConvNeXt-Tiny)
- rishijayanath/parallax-p2-6-bone-fracture-detector: started 2026-10-07 01:00, running (yolo26n, 80 epochs)
- rishijayanath/parallax-p2-3-brain-mri-classifier and parallax-p2-4-brain-tumour-segmenter: queued; Kaggle allows 2 GPU sessions at once, a background loop pushes them when a slot frees

### P3 — Clinical Reasoning & Trust
**Now:**
**Next:**
**Blockers:**
**MedGemma endpoint:** (where it runs, current URL, last health check)

### P4 — Experience & Platform
**Now:** P4.1 to P4.5 done. P4.3 added the orchestrator (`pipeline.run_study`, `core/{context,config,cache}.py`) — see the shared log for that pass's detail. P4.4+P4.5 add:
- `core/ledger.py` (`Ledger`): one global, append-only, hash-chained JSONL (not per-study — matches the single `GET /ledger/verify` endpoint). `append()` hashes `prev_hash + event + payload_hash + ts + actor + study_id`, so tampering any of those fields alone (not just the payload) is caught. `verify()` replays the whole chain.
- `core/store.py` (`StudyStore`): in-memory, per-study bookkeeping for the API — status, accumulated `StageResult`s, final `StudyResult`, an `asyncio.Queue` for live SSE subscribers. **Not persisted**: a server restart loses every in-flight/completed study. That's a known, stated limitation, not an oversight — plan.md's P4.4 Done-when doesn't ask for persistence, and the ledger is the durable record.
- `api/{app,studies,system}.py` (FastAPI + `sse-starlette`): `POST /studies` (multipart upload, runs the pipeline on an executor thread, returns 202 immediately), `GET /studies/{id}/events` (SSE: one `stage` event per `StageResult`, a final `done` with the full `StudyResult`), `GET /studies/{id}`, `POST /studies/{id}/feedback` (the only place a doctor's accept/reject reaches the ledger — D17), `GET /studies/{id}/fhir` / `GET /metrics` / `GET /model-cards` (honest "not available yet" bodies — P3.13/P2.14/P2.13 aren't built, so these never fabricate data), `GET /ledger/verify`.
- `pipeline.run_study` gained one additive parameter, `on_stage_result` (callback fired after every stage, cache hit or miss) — the API uses it to push SSE events and append ledger entries; nothing in `pipeline.py` itself imports the ledger, keeping it usable standalone. The API overwrites `StudyResult.ledger_head` with the real ledger's head after a run; direct `run_study()` callers (tests, future batch tooling) still get `_fold_ledger_head`'s in-memory provisional value, now clearly the "no ledger wired" fallback rather than the only option.
- Verified end to end with a real running server (not just `TestClient`): started `uvicorn medproof.api.app:app`, uploaded a real phantom PNG, confirmed `GET /studies/{id}`, `POST /studies/{id}/feedback`, and `GET /ledger/verify` all behave correctly over real HTTP (ledger `entries` went 2 → 3 after feedback, `ok: true` throughout).

**Honesty note carried over from P4.3**, now doubly true: the generalist/MedGemma stage still doesn't exist (P3.1 `todo`), so "kill MedGemma mid-run" is still unverified as literally stated — proven instead via the same stand-ins (synthetic stages, real brain/skin/bone degradation), now exercised through the HTTP API too, not just `run_study()` directly.
**Next:** P4.6 (web app pages) and P4.7 (WebGL viewer) — first real UI, built against `contracts.ts` and this API.
**Blockers:** none. Known gaps to flag, not blockers: in-memory `StudyStore` isn't persisted (above); the SSE queue fans out to at most one live drainer per study (fine for today's single-viewer-per-study UI, would need rework for multi-viewer); no `make`/`pnpm` binary on this machine so `make setup`/`make dev` are untested end-to-end here (the pieces run fine individually — `uvicorn` was started and curled directly); `ml/` still has no dependency declarations of its own.

## Decisions

Deviations from plan.md, newest last. Format: `date · role · decision · evidence`.

- 2026-10-06 · P1 · Pulled `verify/perturbations.py` forward from P1.11 · shared by the quality-gate fixtures, the stability score and P2.12, so it is defined once.
- 2026-10-06 · P1 · Added `intake/preprocess.py` (not in plan) · one preprocessing path for training notebooks and inference prevents train/serve skew.
- 2026-10-06 · P1 · Blur metric is Laplacian variance divided by intensity variance, computed on a 512 px copy · raw Laplacian variance changes with contrast and resolution; test shows scores within 35% across a 512 to 1024 px resize.
- 2026-10-06 · P1 · CXR tilt is estimated from the rotation that maximises left-right mirror symmetry (with an off-centre shift allowed), not the lung-mask axis · the anatomy segmenter arrives with P1.6; replace then. Only a tilt with symmetry gain of at least 0.03 is reported.
- 2026-10-06 · P1 · PHI scrub keeps PatientAge and PatientSex, blanks names and identifiers, removes free-text descriptions, shifts dates by one random per-study offset that is not recorded, remaps UIDs, drops private tags, and warns on burned-in annotation · the context checks need age and sex; this is a PS3.15 subset, not a certified de-identifier.
- 2026-10-06 · P1 · Quality thresholds are provisional values tuned on a synthetic phantom: the clean phantom passes with zero reasons across 8 seeds, and 100% of noise, blur and contrast degradations at severity 4 and 5 fail the gate · must be refit on real validation images.
- 2026-10-07 · P1 · Wrote the Grad-CAM family (gradcam, gradcam++, hirescam) in `readers/cam.py` instead of using pytorch-grad-cam · fewer heavy dependencies on a slow network, full control for the saliency sanity check (P1.10), and unit tests against a model with a known answer. Default is `gradcam++`, configurable (`MEDPROOF_CXR_CAM`); P1.10 picks the final variant.
- 2026-10-07 · P1 · Reader findings leave as `status="uncertain"`, `prob_calibrated=prob_raw`, `conformal_set=[]`, flag `uncalibrated` · a reader cannot verify itself; faithfulness (P1.9), stability, second read and the status rule upgrade it, and calibration fills the rest. No contract change.
- 2026-10-07 · P1 · One `ImageEvidence` (kind `heatmap`, with `bbox_xyxy`, `heatmap_ref`, `region_name`) per finding; a finding with neither a box nor a heatmap is withheld with a warning, and a flat CAM is flagged `no_localization` · a finding with no image location fails the judges' rule.
- 2026-10-07 · P1 · Changed `cxr_xrv` preprocessing from stretch to centre-crop (see Verified facts) and added `cxr_anatomy` (512 px, same crop). Only P1 code and tests used `cxr_xrv`; P2 is told in the handoff above.
- 2026-10-07 · P1 · Positive threshold 0.5 (each label's own operating point) and tiers high >= 0.8, moderate >= 0.65, else low · provisional until P2 calibration; they live in `readers/cxr_config.py`. Labels are TorchXRayVision's names until P4 supplies `core/vocab.py`.
- 2026-10-07 · P1 · Anatomy pixels inside the heart mask count as heart when naming ("cardiac region"); lung thirds still use each full lung extent · found on the real sample film, where Cardiomegaly was first named "left middle zone".
- 2026-10-07 · P1 · Reader degrades instead of failing: no segmenter gives findings without region names and flag `anatomy_unavailable`; a heart on the image-left gives flag `laterality_uncertain`; model load failure gives a failed `StageResult`.
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
- 2026-10-07 · P2 · BDNeuro-MRI is not an independent external test: 4,238 of its 5,941 images (71%) are near-duplicates (pHash Hamming <= 4, pixel correlation median 0.93 to 1.0 against 0.47 for random same-class pairs) of images in the Kaggle brain dataset. The external test uses only the 1,644 images with no match (`ml/data/splits/bdneuro.csv`, `split == external_test`) · its README claims hospital provenance and exact/near-duplicate removal only within itself.
- 2026-10-07 · P4 · Acked P2's `.gitignore` carve-out and `ml/Makefile.inc` (2026-10-06 entry above) · both already correct; root `Makefile` now `include`s `ml/Makefile.inc` unchanged.
- 2026-10-07 · P4 · G0 contract freeze (P4.2) is purely additive: every existing field/class in `schemas.py` is untouched, confirmed by grepping all of `backend/` for `core.schemas` imports (only `Finding`, `ImageEvidence`, `StageResult` are consumed anywhere today) · no Contract change request needed; nothing P1/P2 built against it changes shape.
- 2026-10-07 · P4 · `StudyContext` (plan.md section 4's stage interface) is deliberately not defined as a class yet; `intake/run.py` already works via duck-typed `ctx.raw_bytes`/`ctx.modality_hint`/`ctx.path` · belongs with the P4.3 orchestrator, where stage ordering and caching actually need a concrete type.
- 2026-10-07 · P4 · Pydantic's auto-generated per-field `"title"` is stripped before writing `contracts/*.schema.json` and before the TS generation step · untitled, each field became its own hoisted top-level TS type (e.g. `Label1`) instead of an inline property; titles aren't part of the contract, field names/types are.
- 2026-10-07 · P4 · `StudyContext` (core/context.py) is a plain dataclass, not pydantic · it carries live `DecodedImage`/numpy data that has no business on a wire-contract model; kept duck-type-compatible with what P1's existing `getattr(ctx, name, default)` stages already read.
- 2026-10-07 · P4 · The orchestrator's `intake` step calls `intake.run_bytes()` directly instead of the generic `intake.run(ctx)` stage wrapper · `run(ctx)` discards the `DecodedImage` it decodes, but the reader stage needs it; `run_bytes`'s own docstring says it returns both specifically so a caller can do this. No change to P1's intake module.
- 2026-10-07 · P4 · Added `diskcache` to `backend/pyproject.toml` dependencies (not dev) for `core/cache.py` · already plan-approved in plan.md section 5.5 (Apache-2.0), logging here for visibility rather than because it needed a CCR.
- 2026-10-07 · P4 · Per-stage timeout uses a daemon-less `ThreadPoolExecutor` with `future.result(timeout=...)`, shut down with `wait=False` on timeout · `signal.alarm` isn't available on Windows (the team develops on Windows laptops and Linux Kaggle/Colab alike); multiprocessing was rejected as pickling `DecodedImage`/reloading a torch model per call costs more than the problem it solves. **Caveat that matters**: this stops the orchestrator from *waiting*, it cannot forcibly kill a CPU-bound native stage (e.g. a blocking torch forward pass) — the thread keeps running in the background and its result is discarded. Correct for stages that raise or block on I/O (HTTP calls to MedGemma/Groq); not a hard kill switch for a hung CPU-bound stage.
- 2026-10-07 · P4 · `StudyResult.ledger_head` is filled by folding `sha256(prev + stage + payload_hash)` over the stage results in-memory (`pipeline._fold_ledger_head`) · explicitly provisional, not the real ledger: no persistence, no append-only JSONL, no `/ledger/verify`. That's P4.5. The fold formula is meant to generalize into the real one, not be thrown away.
- 2026-10-07 · P4 · P4.3's stated Done-when ("kill MedGemma mid-run") is not literally tested: the MedGemma/generalist stage doesn't exist yet (P3.1 `todo`) · verified instead with a synthetic slow/raising stage (mechanism) and the real brain/skin/bone "no reader yet" case (an actual present-day degradation). Flagging this rather than overclaiming the Done-when was met as written.
- 2026-10-07 · P4 · The evidence ledger (P4.5) is one global JSONL, not one per study · matches plan.md's single `GET /ledger/verify` route (no study id in the path); entries carry `study_id` so a future per-study audit view (P4.6) can filter client-side without a new endpoint.
- 2026-10-07 · P4 · `StudyStore` (API-side, in-memory, P4.4) is explicitly not persisted — a server restart loses every study · plan.md's P4.4 Done-when doesn't ask for persistence, and the ledger is the durable record; revisit only if a demo scenario needs studies to survive a restart.
- 2026-10-07 · P4 · Ledger/store are constructor-injected onto `app.state` in `api/app.py`, not module-level singletons like `readers/cxr.py`'s `get_reader()` · the API specifically needs per-test isolation since many pytest tests share one process; a global would leak state between them. `cxr.py`'s singleton pattern is for an expensive model load, a different concern.
- 2026-10-07 · P4 · `GET /studies/{id}/fhir`, `GET /metrics`, `GET /model-cards` return an honest `{"available": false, ...}`/empty body rather than fabricated data, since P3.13/P2.14/P2.13 don't exist yet · consistent with the no-fabrication stance already applied to `pipeline.py`'s placeholders (`ood_score=0.0`, `claims=[]`).
- 2026-10-07 · P4 · The per-study SSE queue fans out to at most one live drainer at a time (a second concurrent subscriber only gets the terminal state once the first drains it) · matches today's single-viewer-per-study UI; true multi-viewer broadcast would need a pub/sub fan-out, not asked for yet.
## Contract change requests

Format: `id · proposer · change · affected roles · P4 ack (yes/no) · applied in commit`.

## Verified facts log

Record anything marked [VERIFY] in plan.md once checked. Format: `date · role · item · result · source`.

- 2026-10-07 · P1 · torchxrayvision 1.5.5 DenseNet `densenet121-res224-all` output · forward() applies sigmoid then operating-point scaling (`op_norm`) when `op_threshs` is set (it is, for every pretrained set), so outputs are probabilities with 0.5 at each label's operating point; labels with a NaN threshold (some chex and mimic sets) output a constant 0.5 · installed `models.py` source.
- 2026-10-07 · P1 · Label set and order of `-all` · 18 labels, identical to `xrv.datasets.default_pathologies`; other weight sets blank some labels with empty strings · installed `models.py` and a real-weights test.
- 2026-10-07 · P1 · Backbone layer for CAM · `model.features` is an `nn.Sequential` ending in `norm5`, applied before the final ReLU, so the captured map can be negative; Grad-CAM++ inverted the map on such inputs until activations were rectified (fixed, regression test) · installed `models.py`, test_cam.
- 2026-10-07 · P1 · Input convention · single channel, range [-1024, 1024], square. The library's transform is `XRayCenterCrop` then `XRayResizer(224)`; the PSPNet raises on non-square input · installed `datasets.py` and `utils.py`. Our `prepare(..., "cxr_xrv")` matches it to within a mean of 25 grey levels on a +-1024 scale and 0.08 in probability on a non-square phantom (resampling filter differs: cv2 INTER_AREA vs skimage).
- 2026-10-07 · P1 · Anatomy model `chestx_det.PSPNet` · input square [-1024,1024] resized to 512, output raw logits (14, 512, 512), target order verified. "Left Lung" is the patient's left lung: on a real NIH chest film (public sample from the torchxrayvision repo, not stored) the "Right Lung" mask centroid was at x=151 and "Left Lung" at x=351 of 512, heart at x=290, matching the standard display with the patient's right on the image left and the film's own L marker on the image right. Lung masks include retrocardiac lung and overlap the heart mask (handled in `zones.name_region`) · one-off real-image run.
- 2026-10-07 · P1 · Download environment · Windows curl failed TLS revocation checks here and the library downloader stalled at 0 bytes, so the two public weight files were fetched with `curl --ssl-no-revoke` (TLS chain still verified, only the revocation lookup skipped) from the official `mlmed/torchxrayvision` v1 release URLs. No checksum is published for them; the weights' sha8 appears in `model_id`.

## Shared log (append-only, newest at the bottom)

<!-- Add entries below this line using the format in plan.md §12.3. Never edit earlier entries. -->

### 2026-10-06 · P1 · WP P1.1, P1.2, P1.3
Did: DICOM/PNG/JPG decode with rescale, windowing and MONOCHROME1 inversion; shared preprocessing specs; eight seeded perturbations; PHI scrub with receipt; quality gate with fix text per failure; intake stage adapter returning StageResult.
State: P1.1 to P1.3 done. Perturbation library done early (P1.11 logic still todo).
Verified: `cd backend && python -m pytest -q` gives 121 passed. Includes a pydicom-built MONOCHROME1 fixture, a PHI sentinel test on the serialized bytes, and six degraded images each giving the right reason.
Next: CXR reader and anatomy zones (P1.6) so P2.7 can calibrate; then router.
Decisions: see Decisions, entries dated 2026-10-06 for P1.
Contract change requests: none.

### 2026-10-07 · P1 · WP P1.6
Did: TorchXRayVision DenseNet121 reader with Grad-CAM family localization (own implementation), PSPNet anatomy masks, patient-side zone naming, contract Finding and ImageEvidence mapping, `run(ctx)` stage. Switched CXR preprocessing to the library's centre-crop convention after reading its source. Fixed a Grad-CAM++ inversion found by the stub tests.
State: P1.6 done. Not yet measured: localization quality (pointing game, box IoU vs RSNA boxes) and faithfulness; those are P2.7 and P1.9. Tiers and the 0.5 threshold are provisional.
Verified: `cd backend && python -m pytest -q` gives 191 passed, 0 skipped, including 5 tests on the real downloaded weights (labels, forward pass, preprocessing parity against the library, end-to-end stage, 14-channel PSPNet). A one-off run on a real public chest film confirmed left/right mask placement and a cardiac-region name for Cardiomegaly; image not stored.
Next: P1.4 router, P1.9 faithfulness, P1.11 flip-rate logic.
Decisions: see Decisions and Verified facts log, entries dated 2026-10-07.
Contract change requests: none.
### 2026-10-07 · P2 · Sonnet 5.5 · WP P2.1, P2.2, P2.3 to P2.6 (notebooks)
Did: resumable checksummed downloader with a pinned lock file; pHash leakage audit and group-aware splits; split files, `reports/leakage.json` and `eval_index.json` for fracatlas and ham10000; cluster-bootstrap CIs and metrics; shared training code; four Kaggle notebooks (brain classifier, brain segmenter, skin classifier, bone detector) plus a builder and a model registry script.
State: P2.1 doing (credential-free sets done; brain, LGG, RSNA wait for the Kaggle token). P2.2 doing (brain, LGG, RSNA splits wait for data). P2.3 to P2.6 blocked: notebooks written and tested, launch needs the Kaggle token and a pushed `p2/data` branch.
Verified: `python -m pytest ml/tests -q` gives 86 passed (brain and segmenter notebooks run end to end on synthetic data; bone notebook smoke-run for one epoch on CPU). Short real skin run on the laptop GPU (3 epochs, 32 s/epoch, same notebook source, throwaway sanity check, not a Kaggle job): official ISIC 2018 test balanced accuracy 0.697 [0.650, 0.745], melanoma recall 0.667. The full 15-epoch run will be higher.
Next: Kaggle token, push `p2/data`, launch the four jobs, build brain/LGG/RSNA splits, then P2.7 (CXR calibration) and the eval harness.
Decisions: see Decisions, entries dated 2026-10-06 for P2.
Contract change requests: none. `.gitignore` carve-out and `ml/Makefile.inc` need a P4 ack (see Decisions).

### 2026-10-07 · P4 · Sonnet 5 · WP P4.1, P4.2 (G0)
Did: root `Makefile` (`setup`, `test`, `smoke`, `contracts`, stub `dev`/`eval`/`demo`; includes `ml/Makefile.inc`); root `pyproject.toml` (ruff + mypy spanning `backend/` and `ml/`); added ruff/mypy to `backend/pyproject.toml`'s dev extra; `.pre-commit-config.yaml` + `.secrets.baseline`; minimal `web/` (Vite + React + TS) so `pnpm install`/`pnpm dev` work and `contracts.ts` has a home. Contract: reviewed `schemas.py` against every actual import in `backend/` and left it untouched (purely additive freeze); added `core/vocab.py` and `core/status_rule.py` (`compute_status`, precedence rejected > discordant > uncertain > verified); `scripts/export_contract.py` → `contracts/{finding,study_result}.schema.json`; `scripts/gen_contracts_ts.mjs` → `web/src/contracts.ts`; one hand-written `StudyResult` fixture per core modality in `contracts/fixtures/`.
State: P4.1 and P4.2 done.
Verified: `cd backend && python -m pytest -q` → 213 passed, 1 skipped, 1 failed (the failure is pre-existing and unrelated: `tests/readers/test_anatomy.py` needs `torchxrayvision`, not installed on this machine). `backend/tests/core` (new) → 28 passed. `cd web && npx tsc -b` and `npx eslint .` → clean. `node scripts/gen_contracts_ts.mjs` regenerates `contracts.ts` byte-identical on a second run.
Next: P4.3 pipeline orchestrator (define `StudyContext` for real here), then P4.4 API + SSE on these fixtures.
Decisions: see Decisions, entries dated 2026-10-07 for P4.
Contract change requests: none (additive only).

### 2026-10-07 · P4 · Sonnet 5 · WP P4.3
Did: `backend/medproof/pipeline.py` (`StageSpec`, `PIPELINE` registry, `run_study()`); `core/context.py` (`StudyContext` dataclass), `core/config.py` (`PipelineConfig`, env-backed), `core/cache.py` (`StageCache` over `diskcache`, keyed by `(input_sha256, stage, version)`). Wired the intake stage via `run_bytes()` (not the generic `run(ctx)` wrapper, which drops the decoded image) and a modality-dispatching reader step (only `cxr` exists; anything else degrades honestly). `_collect_findings` is the first real caller of `status_rule.compute_status` outside its own unit tests. Per-stage timeout via a non-context-managed `ThreadPoolExecutor` (first attempt used `with ThreadPoolExecutor(...)`, whose `__exit__` blocks on the abandoned thread and silently defeated the timeout — caught by the timeout test itself, fixed to call `shutdown(wait=False)` explicitly). Added `diskcache` to `backend/pyproject.toml`. Documented `.env.example`'s previously-missing `MEDPROOF_*` vars.
State: P4.3 done, with one caveat recorded plainly rather than glossed over: the WP's literal Done-when ("kill MedGemma mid-run") isn't testable yet since no MedGemma stage exists (P3.1 `todo`). Verified instead with synthetic slow/raising stages plus the real "no reader for this modality yet" case.
Verified: `cd backend && python -m pytest tests/test_pipeline.py -v` → 8 passed. Full suite `python -m pytest -q` → 221 passed, 1 skipped, 1 failed (same pre-existing unrelated `torchxrayvision` gap as the previous entry, nothing newly broken). `python -m ruff check backend/medproof/pipeline.py backend/medproof/core/` clean.
Next: P4.4, API + SSE, wrapping `run_study()` and streaming each `StageResult` as it completes.
Decisions: see Decisions, entries dated 2026-10-07 for P4.
Contract change requests: none (additive only).

### 2026-10-07 · P4 · Sonnet 5 · WP P4.4, P4.5
Did: `core/ledger.py` (`Ledger`: one global hash-chained append-only JSONL, `append`/`head`/`verify`, hash covers `ts`/`actor`/`event`/`study_id` too, not just the payload); `core/store.py` (`StudyStore`, in-memory, not persisted — stated limitation, see Decisions); `api/{app,studies,system}.py` (FastAPI + `sse-starlette`): `POST /studies`, `GET /studies/{id}/events` (SSE), `GET /studies/{id}`, `POST /studies/{id}/feedback` (the only path to D17's ledger-logged doctor feedback), `GET /studies/{id}/fhir`, `GET /metrics`, `GET /model-cards`, `GET /ledger/verify`. Added one additive parameter to `pipeline.run_study` (`on_stage_result`) so the API can push SSE events and ledger entries per stage without `pipeline.py` itself depending on the ledger. `Makefile`'s `dev` target now actually starts `uvicorn` instead of echoing a stub.
State: P4.4 and P4.5 done. Found and fixed one real bug while writing the SSE test: `StudyStore.stream()`'s original "already-terminal, skip the queue" shortcut silently dropped every buffered stage event for a fast-finishing study (routine with a stub reader finishing before the SSE GET even arrives) — fixed by always draining the queue (it holds every event regardless of when a subscriber connects) and only falling back to the stored terminal state once a queue has been fully drained once already.
Verified: `cd backend && python -m pytest tests/core/test_ledger.py tests/api -v` → 14 passed. Full suite `python -m pytest -q` → 235 passed, 1 skipped, 1 failed (same pre-existing unrelated `torchxrayvision` gap, nothing newly broken). `ruff check` on all new/changed files clean. Manual end-to-end sanity on a real running server (not just `TestClient`): started `uvicorn medproof.api.app:app`, uploaded a real phantom PNG via `curl`, fetched the study, posted feedback, confirmed `GET /ledger/verify` (`entries` 2 → 3, `ok: true` throughout).
Next: P4.6 (web app pages) and P4.7 (WebGL viewer), the first real UI, built against `web/src/contracts.ts` and this API.
Decisions: see Decisions, entries dated 2026-10-07 for P4.
Contract change requests: none (additive only).
