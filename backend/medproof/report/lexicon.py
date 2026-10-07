"""Word lists the firewall and renderer share.

Label terms are the union of P4's core/vocab.py (spoken form: lower case, spaces for underscores) and the
local aliases below (short forms such as "nevus" that a model might type), so a label added to the shared
vocabulary is blocked outside a slot without editing this file.

Everything here is lowercase. Terms are matched on word boundaries after whitespace is collapsed.
"""

from __future__ import annotations

import re

from medproof.core.vocab import VOCAB

CORE_LABEL_TERMS = frozenset(label.lower().replace("_", " ") for labels in VOCAB.values() for label in labels)

# Labels per modality (TorchXRayVision CXR set, HAM10000 classes, brain classes, bone).
CXR_LABELS = (
    "atelectasis", "consolidation", "infiltration", "pneumothorax", "edema", "emphysema", "fibrosis",
    "effusion", "pneumonia", "pleural thickening", "cardiomegaly", "nodule", "mass", "hernia",
    "lung lesion", "fracture", "lung opacity", "enlarged cardiomediastinum",
)
SKIN_LABELS = (
    "melanoma", "melanocytic nevus", "nevus", "basal cell carcinoma", "actinic keratosis",
    "benign keratosis", "dermatofibroma", "vascular lesion",
)
BRAIN_LABELS = ("glioma", "meningioma", "pituitary", "no tumor")
GENERIC_CLINICAL = ("tumor", "tumour", "lesion", "opacity", "cancer", "carcinoma", "malignant", "benign")

# Anatomy, laterality and modality words: they must come from slots, never from the LLM's own text.
REGION_WORDS = (
    "left", "right", "bilateral", "upper", "middle", "lower", "zone", "lobe", "lung", "lungs",
    "heart", "cardiac", "chest", "brain", "skin", "bone", "hand", "wrist", "leg", "hip", "shoulder",
    "knee", "ankle", "foot", "arm", "elbow",
)
MODALITY_WORDS = ("x-ray", "xray", "radiograph", "mri", "ct", "dermoscopy", "dermoscopic", "scan")

VOCAB_TERMS: tuple[str, ...] = tuple(
    sorted(
        {*CORE_LABEL_TERMS, *CXR_LABELS, *SKIN_LABELS, *BRAIN_LABELS, *GENERIC_CLINICAL, *REGION_WORDS, *MODALITY_WORDS},
        key=lambda t: (-len(t), t),
    )
)
VOCAB_RE = re.compile(
    r"(?<![a-z])(?:" + "|".join(re.escape(t) for t in VOCAB_TERMS) + r")(?![a-z])"
)

TIER_PHRASE = {
    "high": "high confidence",
    "moderate": "moderate confidence",
    "low": "low confidence",
    "abstain": "no confident reading",
}
STATUS_PHRASE = {
    "verified": "verified",
    "uncertain": "uncertain",
    "discordant": "discordant between readers",
    "rejected": "rejected",
}
FLAG_PHRASE = {
    "laterality_conflict": "a laterality conflict with the notes",
    "ood": "an image unlike the training data",
    "low_quality": "limited image quality",
    "uncalibrated": "uncalibrated confidence",
    "partial_view": "only part of the image analysed",
}

# Confidence wording. Tier phrases must come from the {fN.tier} slot, not from the LLM.
CONFIDENCE_LITERALS = ("high confidence", "moderate confidence", "low confidence")
STRONG_WORDS = re.compile(
    r"(?<![a-z])(clearly|definitely|certainly|certain|confirmed|confirms|evident|obvious|obviously|"
    r"undoubtedly|proves|demonstrates|consistent with|diagnostic of)(?![a-z])"
)
HEDGE_WORDS = re.compile(r"(?<![a-z])(possibly|possible|may|might|perhaps|maybe|could)(?![a-z])")
SUGGEST_WORDS = re.compile(r"(?<![a-z])(consider|suggest|suggests|likely|probable|probably)(?![a-z])")
REVIEW_WORDS = re.compile(r"disagree|review")

BANNED_PHRASES = (
    "the patient has", "the patient had", "diagnosis is", "diagnosed with", "diagnosis of", "definitely",
    "certainly", "rule out", "rules out", "no abnormality", "normal study", "recommend treatment",
    "prescribe", "you have",
)

# Instruction-like text that must never be echoed from a note.
INSTRUCTION_RE = re.compile(
    r"ignore (?:all |any |the )?(?:previous|prior|above)|disregard|system\s*:|assistant\s*:|"
    r"you must|report no findings|do not mention|print your|system prompt|api key|cite (?:te|ie|f)_?\d+|"
    r"</?\s*note_data|as an ai|new instructions|```",
    re.IGNORECASE,
)
WITHHELD = "[note text withheld: flagged]"
