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
- 2026-10-06 · P1 · Python 3.12 venv with pip instead of 3.11 with uv (uv not installed here); `backend/pyproject.toml` is minimal for the platform owner to absorb · no `make smoke` exists yet, so verification is `python -m pytest` in `backend/`.
- 2026-10-06 · P1 · `core/schemas.py` is a verbatim copy of the section 4 sketch because the contract was not yet committed · platform owner replaces it; P1 code only uses `StageResult`.

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
