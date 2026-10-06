"""Settings for the modality router. Model id, paths, thresholds and prompts live here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

CLASSES = ("cxr", "brain_mri", "skin_dermoscopy", "bone_xray", "other")
BODY_PARTS = ("hand", "leg", "hip", "shoulder")  # FracAtlas body-part labels (bone X-ray only)

# Zero-shot prompt sets. Averaged per class, so more prompts make a steadier prototype.
PROMPTS: dict[str, tuple[str, ...]] = {
    "cxr": (
        "a frontal chest X-ray",
        "a chest radiograph showing the lungs and heart",
        "a posteroanterior chest x-ray",
    ),
    "brain_mri": (
        "a brain MRI slice",
        "an axial magnetic resonance image of the head",
        "a T1-weighted MRI scan of the brain",
    ),
    "skin_dermoscopy": (
        "a dermoscopy image of a skin lesion",
        "a dermatoscopic close-up of a mole",
        "a magnified photograph of a pigmented skin lesion",
    ),
    "bone_xray": (
        "an X-ray of a bone",
        "a radiograph of a hand or wrist",
        "a musculoskeletal radiograph showing bones and joints",
    ),
    "other": (
        "a photograph of an everyday object",
        "a natural photograph of an animal or a scene",
        "a CT scan",
        "a retinal fundus photograph",
        "a histopathology slide",
        "a drawing or a screenshot",
    ),
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class RouterConfig:
    model_id: str = field(default_factory=lambda: _env("MEDPROOF_ROUTER_MODEL", "google/medsiglip-448"))
    probe_path: str = field(default_factory=lambda: _env("MEDPROOF_ROUTER_PROBE", "artifacts/router_probe.npz"))
    cache_dir: str = field(default_factory=lambda: _env("MEDPROOF_EMBED_CACHE", "artifacts/embeddings"))
    device: str = field(default_factory=lambda: _env("MEDPROOF_DEVICE", "cpu"))
    spec: str = "router_medsiglip"  # preprocessing spec in intake.preprocess

    min_confidence: float = 0.80  # probe below this falls back to zero-shot
    zero_shot_floor: float = 0.50  # zero-shot below this gives "other" plus router_uncertain
    trust_confidence: float = 0.60  # intake only trusts a routed modality at or above this
    prompts: dict = field(default_factory=lambda: dict(PROMPTS))
    classes: tuple = CLASSES
