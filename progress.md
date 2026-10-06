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
| P2.2 | Leakage audit + group splits | done | main | 2026-10-07 | all six datasets audited; reports/leakage.json |
| P2.3 | Brain classifier (Kaggle) | done | main | 2026-10-07 | done on Kaggle (second account); registered |
| P2.4 | Brain segmenter (Kaggle) | done | main | 2026-10-07 | done on Kaggle (second account); registered |
| P2.5 | Skin classifier (Kaggle) | done | main | 2026-10-07 | official-test BMA 0.680 [0.632, 0.728]; registered |
| P2.6 | Bone detector (Kaggle) | doing | main | 2026-10-07 | v2 running on Kaggle after a fix |
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
| P3.1 | MedGemma service + batch reads | doing | p3/medgemma-groq | 2026-10-07 | real read verified on the local RTX 4060 (bf16 4-bit, 3.2 GiB, 5.5 s/read after a 23 s load); Kaggle/Colab notebooks untested; eval batch reads running (resumable) |
| P3.2 | Generalist reader + concordance | doing | p3/medgemma-groq | 2026-10-07 | client, label mapping, concordance, kappa done and tested; reports/concordance.json waits for the batch reads; specialist-vs-MedGemma kappa waits for P2/P1 weights |
| P3.3 | Groq pool | done | p3/medgemma-groq | 2026-10-07 | 27 tests with a mocked Groq; live call confirmed on gpt-oss-20b and 120b; Prompt Guard score() added |
| P3.4 | Synthetic note generator | done | p3/medgemma-groq | 2026-10-07 | ml/data/gen_notes.py, 200 notes: 60 contradictions, 48 attacks in 8 families, 12 benign look-alikes, clean twins; notes_v1.jsonl committed |
| P3.5 | Fact extraction with spans | doing | p3/medgemma-groq | 2026-10-07 | code and tests done; held-out numbers in reports/p3_context_test.json |
| P3.6 | Injection guard | doing | p3/medgemma-groq | 2026-10-07 | regex + Prompt Guard + LLM layers done and tested; held-out numbers in reports/p3_context_test.json |
| P3.7 | Contradiction rules | done | p3/medgemma-groq | 2026-10-07 | P/R 1.00/1.00 on gold facts (templated notes, so it shows the rules fire, not that they generalise); end-to-end with extracted facts not yet measured |
| P3.8 | Retrieval / precedents | todo | | | |
| P3.9 | Slot-filled report | done | p3/medgemma-groq | 2026-10-07 | slot renderer, frames, model-chooses-frame drafts with offline template fallback; live gpt-oss-120b call verified |
| P3.10 | Hallucination firewall | done | p3/medgemma-groq | 2026-10-07 | R1-R12; 100% of 100+ planted bad claims blocked, 0 of 10 good claims blocked; R5 treats faithful=None as unverified until P1.9 |
| P3.11 | Entailment judge | done | main | 2026-10-07 | 60 labelled sentences: first pass 0.917; after fixing what the judge is shown, 59/60 on a fresh variant (kept 29/30 true, caught 30/30 defective) and 60/60 on the tuned one; judge outage leaves claims explicitly unchecked |
| P3.12 | Voice (Whisper) | done | main | 2026-10-07 | real round trip: synthesized dictation -> Whisper on Groq -> guard -> facts; spoken injection flagged and kept out of the facts |
| P3.13 | FHIR export | done | main | 2026-10-07 | collection Bundle (DiagnosticReport + Observations) validated with fhir.resources R4B; quotes redacted unless requested |

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
- rishijayanath/parallax-p2-5-skin-lesion-classifier: 2026-10-07 01:00 to 01:17, COMPLETE. Official ISIC test balanced accuracy 0.689 [0.640, 0.738]; output not pulled (Kaggle rate limit), superseded by the rerun below
- rishijayanath/parallax-p2-6-bone-fracture-detector: v1 trained 80 epochs (val mAP50 0.453 on 380 images) then failed at prediction export on truncated JPEGs; fixed in commit 8539baf; v2 relaunched 2026-10-07 02:10
- johannshonigeorge/parallax-p2-5-skin-lesion-classifier: rerun COMPLETE 2026-10-07 02:26, pulled, registered as skin_cls@a217517b. Official ISIC 2018 test balanced accuracy 0.680 [0.632, 0.728], melanoma recall 0.637, melanoma AUROC 0.924; the two runs differ by about 0.01 (GPU nondeterminism)
- johannshonigeorge/parallax-p2-3-brain-mri-classifier: COMPLETE 2026-10-07, pulled, registered as brain_cls@d351f062. Leakage-free test accuracy 0.940 [0.921, 0.958]
- johannshonigeorge/parallax-p2-4-brain-tumour-segmenter: COMPLETE, pulled, registered as brain_seg@d351f062. Mean per-patient Dice 0.831 [0.705, 0.891] over 9 test groups. A duplicate of this job also ran on rishijayanath (the launcher fired before I stopped it); its output is unused

### P3 — Clinical Reasoning & Trust
**Now:** MedGemma eval batch reads are running on the local RTX 4060 (two processes, one per dataset) (`python -m services.medgemma.batch --dataset fracatlas|ham10000`, resumable, fixed random order so any prefix is a random sample); afterwards `python -m ml.eval_p3.eval_second_reader` writes `reports/concordance.json`.
**Next:** P3.8 retrieval (needs P1's MedSigLIP embedding fn and the train split); end-to-end context stage (guard -> extract -> contradictions) measured on extracted instead of gold facts; skin specialist-vs-MedGemma result and trust-signal check once the HAM10000 reads finish; wire stages into P4's pipeline once `StudyContext` exists.
**Blockers:** none. `StudyContext` is not defined yet, so my stages take plain arguments (`StudyView`, facts, `Demographics`); P4 can adapt them.
**MedGemma endpoint:** local only. `uvicorn services.medgemma.server:app --port 8001` on the RTX 4060 (nf4 weights, bf16 compute, 3.2 GiB VRAM, about 5.5 s per read after a 23 s load). Kaggle/Colab notebooks written but not run.
**For P1:** `report/firewall.py::is_supported` treats `faithful=None` as unverified; set `faithful` (P1.9) and R5 tightens automatically. `readers/generalist.py` reads `DecodedImage.display` and `.sha256`; `verify/concordance.apply(findings, read)` fills `Finding.second_read` and adds flags only, never `status`.
**For P2 / P4 (important):** the second reader is weak on bone: on a random 403-image prefix of the FracAtlas batch MedGemma 4B (4-bit, zero-shot) found about 19% of fractures (sensitivity 0.19, 95% CI 0.11-0.29) at 99.4% specificity (kappa 0.27, 0.15-0.38); the missed reads literally say "No obvious fractures are visible". A `discordant` status driven by `second_reader_disagrees` would downgrade most true bone positives. Per plan 9.4, validate the flag (D13) per modality before letting it downgrade anything; until then treat it as an audit flag for bone. Numbers will be in `reports/concordance.json`.
**For P4 (stages ready to wire, in this order):** `context.voice.process_dictation` (optional audio) -> `context.injection_guard.InjectionGuard.check` -> `context.extract.extract` -> `context.contradictions.check` -> `report.drafts.draft_claims` (includes the firewall) -> `report.entailment.judge` -> `report.fhir.build_bundle`. A report consumer must show only claims with `blocked_reason is None`; a claim removed by the judge keeps its `rendered` text so the audit view can strike it through. `build_bundle` writes only reportable claims and redacts note quotes unless `include_quotes=True`.
**For P4:** `report.drafts.draft_claims(view, pool)` returns claims + a `StageResult(stage="report")`; with `pool=None` or Groq down it returns the template-only report and says so in `warnings`. `context.contradictions.check(...)` returns findings with `TextEvidence` attached (one `te_n` id per finding/fact pair) plus `missing_context` prompts. The status rule can count `polarity == "supports"` text evidence.

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
- 2026-10-07 · P2 · Brain classifier headline: the Kaggle benchmark inflation is measured inside one model. Model A scores 0.942 on the original Testing folder, 0.903 once the 727 images with a duplicate in Training are removed, and 0.865 once the 1,051 images with a same-scan neighbour are removed (n=549). The across-model difference A minus B (0.002, CI -0.020 to 0.026) shows no inflation and is confounded by different models and test sets, so it must not be quoted · B (leakage-free group split, test n=563) scores 0.940 [0.921, 0.958].
- 2026-10-07 · P2 · Two Kaggle accounts: the first account's concurrent GPU limit of 2 stalled the brain jobs, so they ran on a second existing account (johannshonigeorge, owner agreed) · plan.md section 14 lists several accounts as the fallback; its token is kept in the gitignored `.env.kaggle2`.
- 2026-10-06 · P3 · Slot `frame` enum on top of free templates, so the "Doctor, consider..." framing is structural · docs/report_slots.md
- 2026-10-06 · P3 · Firewall rule R4 binds cited evidence to the finding named in a slot; adds a mandatory false-block test on good claims · docs/report_slots.md
- 2026-10-06 · P3 · MedGemma service returns boxes on a normalized 0..1000 grid; pixel conversion happens in the reader (P3.2) · services/medgemma/README.md
- 2026-10-06 · P3 · Groq pool adds RPM and daily counters, a total deadline, and `PoolResult.source` (live/cache/fallback) so degradation shows up in StageResult warnings · backend/tests/llm/test_groq_pool.py
- 2026-10-06 · P3 · Pool defaults to `json_object` plus schema text in the system prompt, not strict `json_schema`; unsupported params (`reasoning_effort`, `response_format`) are dropped once on a 400 · live call succeeded without needing the drop
- 2026-10-06 · P3 · Real-GPU smoke is a separate script (`python -m services.medgemma.smoke`); CI tests use a fake backend · services/medgemma/
- 2026-10-06 · P3 · Added `.cache/` and `ml/artifacts/medgemma_reads/` to .gitignore (LLM cache can hold note text; cached reads are large) · plan.md 11.3

- 2026-10-07 · P3 · MedGemma runs in bf16 compute, not fp16: fp16 returned an empty reply on the RTX 4060 (Gemma-family activations overflow). The loader picks bf16 when the GPU supports it, else fp32 · services/medgemma/loader.py, smoke log
- 2026-10-07 · P3 · The second reader is asked for a short FINDINGS/IMPRESSION report with a research-evaluation framing, not strict JSON: the JSON-only prompt gave looping labels, refusals and reasoning-only output. Labels are derived deterministically from the text (negation and hedging aware) in `verify/report_labels.py`; boxes from MedGemma are not requested yet · services/medgemma/prompts, 44 label-mapping tests
- 2026-10-07 · P3 · Injection guard is three layers (regex, Prompt Guard per sentence at 0.9, gpt-oss-20b classifier). On 28 fresh attacks: regex 4, Prompt Guard 4 (7 at 0.5), classifier 25, all three 27, with 1 false positive in 15 benign. The regex 15/15 on the corpus is in-sample and must not be quoted · reports/p3_guard_blind.json
- 2026-10-07 · P3 · Gold quote conventions in gen_notes were changed after reading dev-split extraction errors (medication names without 'on', no compound symptoms, foreign negators kept, foreign age fact added). The test split had not been scored · commit "fix(notes): align gold quote conventions"
- 2026-10-07 · P3 · Differential injection test measures (1) facts overlapping injected text and (2) F1 against gold on injected notes vs their clean twins, not identical fact sets, which only reflect run-to-run variance (5/15 identical) · ml/eval_p3/eval_context.py
- 2026-10-07 · P3 · TextEvidence ids are unique per (finding, fact) pair so each id has one owner, as the firewall's evidence-owner rule (R4) assumes · context/contradictions.py
- 2026-10-07 · P3 · Report claims are drafted as frames chosen by the model (ids only, no labels or quotes in its input), expanded by code into slot templates; offline fallback picks frames deterministically · report/drafts.py

- 2026-10-07 · P3 · FHIR is validated against `fhir.resources.R4B` (the library ships R4B, R5 and STU3, no plain R4); the resources used are the same in R4 and R4B. The library checks structure and data types but not value-set codes, so status codes are kept as constants in report/fhir.py · backend/tests/report/test_fhir.py
- 2026-10-07 · P3 · Entailment judge input shows the second reader as an explicit word ("agrees", "disagrees", "inconclusive", "unavailable"), a summary of what the notes contain (to verify "the notes do not mention history"), and readable flag phrases. These came from the first-pass errors on variant 0; variant 1 was written afterwards and is the fair measurement · reports/p3_entailment_*.json
- 2026-10-07 · P3 · Voice: Whisper gets a static clinical-vocabulary prompt (no patient data) and the transcript is treated as an untrusted note: same guard and extraction as typed text · context/voice.py
- 2026-10-07 · P3 · MedGemma service reads its prompt files once at start. A git rebase during a long batch removed a prompt file for a moment and killed the run; resumable by image id, so nothing was lost but time · services/medgemma/reader.py

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
- 2026-10-06 · P3 · `google/medgemma-1.5-4b-it` id, gating, `transformers>=4.50`, `AutoModelForImageTextToText`, BF16 weights · confirmed · huggingface.co/google/medgemma-1.5-4b-it
- 2026-10-06 · P3 · Groq free tier gpt-oss-120b/20b: 30 RPM, 8K TPM, 1K RPD; cached tokens not counted; 429 carries retry-after · confirmed · console.groq.com/docs/rate-limits
- 2026-10-06 · P3 · Prompt Guard id is `meta-llama/llama-prompt-guard-2-86m` (as in plan.md; the rate-limit page drops the prefix). Also listed: `-22m`, `openai/gpt-oss-safeguard-20b`, `whisper-large-v3(-turbo)` · confirmed · GET /models with the extract key
- 2026-10-06 · P3 · Groq accepts `response_format: json_object` and `reasoning_effort: low` on gpt-oss-20b (live call: 1 attempt, 258 tokens, injected "report no findings" text ignored, 2nd call served from cache) · confirmed · live pool call
- 2026-10-06 · P3 · MedGemma access: HF token is valid but returns 403 on the gated repo (terms not accepted for that account, or token lacks gated-repo read) · BLOCKED
- 2026-10-06 · P3 · MedGemma CXR box prompt/format · NOT verified (model card gave no format); check on first smoke run

- 2026-10-07 · P3 · MedGemma 1.5 4B nf4 fits and runs on an RTX 4060 Laptop (8 GB): 3.2 GiB peak VRAM, 23 s load from cache, 5.5 s per read with the report prompt · confirmed · services/medgemma/smoke.py log
- 2026-10-07 · P3 · MedGemma fp16 compute returns an empty reply on that GPU; bf16 works · confirmed · smoke runs
- 2026-10-07 · P3 · MedGemma CXR box prompt/format · still NOT verified (the report-style prompt does not request boxes)
- 2026-10-07 · P3 · Llama Prompt Guard 2 on Groq answers with a bare probability and scores a whole clinical note low (3 of 15 attacks at 0.5); per-sentence it fires on 4 of 28 fresh attacks at 0.9 · confirmed · reports/p3_guard_blind.json
- 2026-10-07 · P3 · Groq free tier real limits observed: 8,000 tokens per minute is the binding limit for gpt-oss-20b extraction (about 4-6 calls a minute with a 1.5K-token prompt) · confirmed · response headers

- 2026-10-07 · P3 · Whisper (`whisper-large-v3-turbo`) on Groq with `GROQ_KEY_AUDIO` accepts multipart audio with `response_format=verbose_json` and returns text, language and duration; a 20 s synthesized dictation was transcribed correctly including numbers · confirmed · ml/eval_p3/voice_roundtrip.py
- 2026-10-07 · P3 · fhir.resources 8.3.0 does not enforce value-set bindings (an invalid `status` string validates); required fields and date formats are enforced · confirmed · test_fhir.py

## Shared log (append-only, newest at the bottom)

<!-- Add entries below this line using the format in plan.md §12.3. Never edit earlier entries. -->

### 2026-10-06 · P1 · WP P1.1, P1.2, P1.3
Did: DICOM/PNG/JPG decode with rescale, windowing and MONOCHROME1 inversion; shared preprocessing specs; eight seeded perturbations; PHI scrub with receipt; quality gate with fix text per failure; intake stage adapter returning StageResult.
State: P1.1 to P1.3 done. Perturbation library done early (P1.11 logic still todo).
Verified: `cd backend && python -m pytest -q` gives 121 passed. Includes a pydicom-built MONOCHROME1 fixture, a PHI sentinel test on the serialized bytes, and six degraded images each giving the right reason.
Next: CXR reader and anatomy zones (P1.6) so P2.7 can calibrate; then router.
Decisions: see Decisions, entries dated 2026-10-06 for P1.
### 2026-10-06 · P3 · Sonnet 5.5 · WP P3.1, P3.3, P3.4 (schema), P3.9/P3.10 (design)
Did: Groq pool (cache, RPM/TPM budgeter, retry-after backoff, repair retry, fallback); MedGemma read service (loader, API, batch, Kaggle/Colab notebooks); note schema with injection taxonomy; slot grammar and firewall spec.
State: P3.3 done. P3.1 code done, real-model smoke NOT run (no HF_TOKEN here). P3.4 schema done, generator todo.
Verified: `.venv/Scripts/python -m pytest ml backend services -q` -> 48 passed, 1 skipped (the gpu-marked smoke); `mypy --strict backend/medproof/llm` clean. `make smoke` does not exist yet (P4.1).
Next: run smoke on the local RTX 4060 once HF_TOKEN is set; check CXR box format; gen_notes.py; P3.2.
Decisions: see Decisions section (7 entries).
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

### 2026-10-07 · P3 · Sonnet 5.5 · WP P3.1, P3.2, P3.4-P3.7, P3.9, P3.10
Did: slot renderer + firewall + model-chosen frames with offline fallback; seeded 200-note generator; injection guard (regex, per-sentence Prompt Guard, classifier) and span-grounded extraction; contradiction rules; report-label mapping, MedGemma client, concordance; real MedGemma reads on the local GPU; eval scripts that recompute from cached output.
State: P3.3, P3.4, P3.7, P3.9, P3.10 done. P3.1/P3.2 doing (eval batch reads running; concordance.json is partial until they finish). P3.5/P3.6 measured below.
Verified: `.venv/Scripts/python -m pytest backend ml/tests/test_notes_schema.py ml/tests/test_gen_notes.py ml/tests/test_eval_second_reader.py ml/tests/test_eval_contradictions.py services -q` -> all pass (ml/ tests that need pandas/sklearn/torch are P2's and were not run in my light venv).
Numbers (62 held-out test-split notes, live Groq, cached): extraction exact-span F1 0.919 (P 0.922, R 0.917), lenient 0.973, 0 failures; weakest types laterality 0.79, history 0.80, device 0.82. Guard on the corpus 15/15 attacks, 0/47 false positives (regex is in-sample). Guard on 28 fresh attacks / 15 fresh benign: regex 4/28, Prompt Guard@0.9 4/28, classifier 25/28 (1 FP), all three 27/28 (1 FP). Differential: 0 facts overlap the injected text; F1 0.954 on injected notes vs 0.936 on their clean twins. Contradiction rules P/R 1.00/1.00 on gold facts (templated notes: rules fire as designed, not a generalisation claim). Firewall: 104 planted bad claims all blocked, 10 good claims none blocked. Second reader on bone (275-read prefix, reports/concordance.json): sensitivity 0.19 (0.09-0.32), specificity 0.995, kappa 0.27 (0.12-0.43).
Next: finish the batch reads and write reports/concordance.json; retrieval (needs P1's embeddings); entailment judge; end-to-end context stage on extracted facts; wire into P4's pipeline.
Decisions: see Decisions (7 entries dated 2026-10-07).
Contract change requests: none.

### 2026-10-07 · P3 · Sonnet 5.5 · WP P3.11, P3.12, P3.13
Did: entailment judge (second wall behind the firewall); Whisper transcription in the Groq pool plus voice dictation through the guard and extraction; FHIR bundle export. Pulled P2's skin-classifier push and wrote the specialist-vs-MedGemma comparison from its saved official-test predictions (agreement, kappa, and whether agreement predicts that the specialist is right); it runs when the HAM10000 reads are in. Found and fixed a crash in the long MedGemma batch (see Decisions).
State: P3.11, P3.12, P3.13 done and pushed. HAM10000 reads have only just started, so the skin comparison has no numbers yet; FracAtlas reads about 70% done.
Verified: pytest backend + ml/tests (mine) + services: 560+ passed, 5 skipped (the gpu-marked smoke and P2 tests that need torch/sklearn). Entailment: first pass 55/60; fixed judge 59/60 on the fresh variant (29/30 true sentences kept, 30/30 defective caught), 60/60 on the tuned variant; false rejections are the judge's main error (about 2-3%). Voice: real round trip OK. FHIR: bundle validates; 11 tests.
Next: run `python -m ml.eval_p3.eval_second_reader` when the reads finish, then report whether specialist agreement predicts specialist correctness (D13 input for P2); retrieval needs P1's embedding function; wire stages into P4's pipeline.
Decisions: see Decisions (4 entries dated 2026-10-07 added).
Contract change requests: none.
