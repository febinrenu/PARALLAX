"""Controlled label vocabulary per modality (plan.md section 4: "label: str # controlled
vocabulary per modality"). `Finding.label` for a given modality should come from here so the
UI, model cards and the status rule can treat labels as a closed set instead of free text.

Readers are the source of truth for what they actually emit; a reader's own label list always
wins over this file in a mismatch (e.g. `CxrReader.labels` loads straight from the installed
torchxrayvision model). Treat this module as documentation + a drift check, not a hard gate.
"""

from __future__ import annotations

from medproof.core.schemas import Modality

# Mirrors torchxrayvision 1.5.5's `datasets.default_pathologies` (18 labels, checked against the
# installed package 2026-10-07). P1's `CxrReader.labels` reads the list from the model at
# runtime and stays authoritative if a future release changes it.
CXR_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)

# Provisional: plan.md section 2.1, 4 classes (Brain Tumor MRI Dataset). Owner: P2 (P2.3).
BRAIN_MRI_LABELS: tuple[str, ...] = (
    "glioma",
    "meningioma",
    "pituitary",
    "no_tumor",
)

# Provisional: plan.md section 9.2, HAM10000's 7 diagnostic classes. Owner: P2 (P2.5).
SKIN_DERMOSCOPY_LABELS: tuple[str, ...] = (
    "melanoma",
    "melanocytic_nevus",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "benign_keratosis",
    "dermatofibroma",
    "vascular_lesion",
)

# Provisional: FracAtlas is a binary fracture detector (plan.md section 5.1). Owner: P2 (P2.6).
BONE_XRAY_LABELS: tuple[str, ...] = (
    "fracture",
)

VOCAB: dict[Modality, tuple[str, ...]] = {
    "cxr": CXR_LABELS,
    "brain_mri": BRAIN_MRI_LABELS,
    "skin_dermoscopy": SKIN_DERMOSCOPY_LABELS,
    "bone_xray": BONE_XRAY_LABELS,
    "other": (),
}


def labels_for(modality: Modality) -> tuple[str, ...]:
    """The known label set for a modality; empty for `other` (generalist reader only)."""
    return VOCAB[modality]
