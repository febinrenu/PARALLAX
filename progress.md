# progress.md — Parallax live state

> Single source of truth for what is happening right now. Every session reads this after CLAUDE.md and plan.md.
> **Rules:** edit only your own status rows and your own role section. The shared log at the bottom is append-only: add new entries at the very end, never edit old ones. Format is in plan.md §12.3. `.gitattributes` sets `merge=union` for this file.

## Team

| Role | Person | Machine / GPU | Groq key variable |
|---|---|---|---|
| P1 Imaging Core | | | GROQ_KEY_AUDIO |
| P2 Data, Training & Validation | | | GROQ_KEY_JUDGE |
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
| P2.1 | Dataset downloads + manifests | todo | | | |
| P2.2 | Leakage audit + group splits | todo | | | |
| P2.3 | Brain classifier (Kaggle) | todo | | | |
| P2.4 | Brain segmenter (Kaggle) | todo | | | |
| P2.5 | Skin classifier (Kaggle) | todo | | | |
| P2.6 | Bone detector (Kaggle) | todo | | | |
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
**Now:**
**Next:**
**Blockers:**
**Kaggle jobs:** (notebook slug, started, status, artifact hash)

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
- 2026-10-07 · P1 · Wrote the Grad-CAM family (gradcam, gradcam++, hirescam) in `readers/cam.py` instead of using pytorch-grad-cam · fewer heavy dependencies on a slow network, full control for the saliency sanity check (P1.10), and unit tests against a model with a known answer. Default is `gradcam++`, configurable (`MEDPROOF_CXR_CAM`); P1.10 picks the final variant.
- 2026-10-07 · P1 · Reader findings leave as `status="uncertain"`, `prob_calibrated=prob_raw`, `conformal_set=[]`, flag `uncalibrated` · a reader cannot verify itself; faithfulness (P1.9), stability, second read and the status rule upgrade it, and calibration fills the rest. No contract change.
- 2026-10-07 · P1 · One `ImageEvidence` (kind `heatmap`, with `bbox_xyxy`, `heatmap_ref`, `region_name`) per finding; a finding with neither a box nor a heatmap is withheld with a warning, and a flat CAM is flagged `no_localization` · a finding with no image location fails the judges' rule.
- 2026-10-07 · P1 · Changed `cxr_xrv` preprocessing from stretch to centre-crop (see Verified facts) and added `cxr_anatomy` (512 px, same crop). Only P1 code and tests used `cxr_xrv`; P2 is told in the handoff above.
- 2026-10-07 · P1 · Positive threshold 0.5 (each label's own operating point) and tiers high >= 0.8, moderate >= 0.65, else low · provisional until P2 calibration; they live in `readers/cxr_config.py`. Labels are TorchXRayVision's names until P4 supplies `core/vocab.py`.
- 2026-10-07 · P1 · Anatomy pixels inside the heart mask count as heart when naming ("cardiac region"); lung thirds still use each full lung extent · found on the real sample film, where Cardiomegaly was first named "left middle zone".
- 2026-10-07 · P1 · Reader degrades instead of failing: no segmenter gives findings without region names and flag `anatomy_unavailable`; a heart on the image-left gives flag `laterality_uncertain`; model load failure gives a failed `StageResult`.
- 2026-10-06 · P1 · Python 3.12 venv with pip instead of 3.11 with uv (uv not installed here); `backend/pyproject.toml` is minimal for the platform owner to absorb · no `make smoke` exists yet, so verification is `python -m pytest` in `backend/`.
- 2026-10-06 · P1 · `core/schemas.py` is a verbatim copy of the section 4 sketch because the contract was not yet committed · platform owner replaces it; P1 code only uses `StageResult`.

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
