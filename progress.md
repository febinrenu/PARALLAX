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
| P1.1 | Decode PNG/JPG/DICOM + hashing | todo | | | |
| P1.2 | DICOM PHI scrub | todo | | | |
| P1.3 | Quality gate | todo | | | |
| P1.4 | Router (MedSigLIP probe) | todo | | | |
| P1.5 | OOD score | todo | | | |
| P1.6 | CXR reader + anatomy zones | todo | | | |
| P1.7 | Bone reader wrapper | todo | | | |
| P1.8 | Brain + skin inference wrappers | todo | | | |
| P1.9 | Faithfulness test (D1) | todo | | | |
| P1.10 | Saliency sanity check (D2) | todo | | | |
| P1.11 | Stability score (D5) | todo | | | |
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
**Now:**
**Next:**
**Blockers:**

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

## Contract change requests

Format: `id · proposer · change · affected roles · P4 ack (yes/no) · applied in commit`.

## Verified facts log

Record anything marked [VERIFY] in plan.md once checked. Format: `date · role · item · result · source`.

## Shared log (append-only, newest at the bottom)

<!-- Add entries below this line using the format in plan.md §12.3. Never edit earlier entries. -->
