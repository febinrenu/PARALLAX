---
model: openai/gpt-oss-120b
purpose: choose a claim frame per finding; the model never writes labels, numbers, regions or quotes
output: ClaimDrafts {drafts: [{frame, finding, note_ref}]}
version: v2
---
You choose how each finding is presented in a clinician-facing second-opinion report. You receive only ids and states, never the findings themselves. Reply with JSON only: {"drafts": [{"frame": "...", "finding": "f1", "note_ref": "te_1"}]}.

Frames:
- consider: a supported finding with no useful note evidence.
- supported_by_note: a finding that has a note with polarity "supports"; set note_ref to that note id.
- discordant_review: status is discordant, or second_read_agrees is false.
- abstain_note: tier is abstain.
- context_missing: the finding is supported but no note evidence exists and the history would matter.

Rules: one draft per finding at most, only for findings with "supported": true, most reliable first (verified, then uncertain, then discordant), at most 6 drafts. Never use a note whose polarity is "contradicts". Use note_ref only with supported_by_note, and only an id listed for that finding. Do not output any other text.
