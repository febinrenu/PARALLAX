"""Intake stage: decode, scrub, quality-check. Returns a StageResult even on failure."""

from __future__ import annotations

import time
from pathlib import Path

from medproof.core.schemas import StageResult
from medproof.intake import decode, phi, quality
from medproof.intake.router_config import RouterConfig

_TRUST_CONFIDENCE = RouterConfig().trust_confidence


def run_bytes(
    raw: bytes, modality_hint: str | None = None, router=None
) -> tuple[StageResult, decode.DecodedImage | None]:
    """Process an upload. The decoded image is returned alongside the StageResult for later stages.

    Without a `modality_hint`, an optional `router` (anything with `.predict(DecodedImage)`) decides
    the modality, which switches modality-specific quality checks. A router failure never fails intake.
    """
    t0 = time.perf_counter()

    def ms() -> int:
        return int((time.perf_counter() - t0) * 1000)

    try:
        img = decode.load_image(raw)
    except decode.DecodeError as exc:
        return StageResult(stage="intake", ok=False, ms=ms(), payload={"error": str(exc)}, warnings=[str(exc)]), None
    warnings = list(img.warnings)
    payload: dict = {"image": img.meta()}

    if img.source_format == "dicom":
        try:
            _, receipt = phi.scrub_bytes(raw)
            payload["phi_scrub"] = receipt.to_dict()
            if receipt.burned_in_annotation:
                warnings.append("DICOM reports burned-in annotation: pixels were not scrubbed")
        except Exception as exc:  # scrub failure must not lose the study, but must be visible
            warnings.append(f"PHI scrub failed: {type(exc).__name__}")
            payload["phi_scrub"] = {"error": "scrub_failed"}

    modality = modality_hint
    if modality_hint is None and router is not None:
        try:
            routed = router.predict(img)
            payload["router"] = routed.to_dict()
            warnings.extend(routed.warnings)
            if routed.confidence >= _TRUST_CONFIDENCE:
                modality = routed.modality
            else:
                warnings.append(f"low router confidence ({routed.confidence:.2f}): modality-specific checks skipped")
        except Exception as exc:  # the router must never take intake down with it
            payload["router"] = {"error": "router_failed"}
            warnings.append(f"router failed ({type(exc).__name__}): modality unknown")
    report = quality.assess(img, modality)
    payload["quality"] = report.to_dict()
    for r in report.reasons:
        warnings.append(f"{r.code}: {r.fix}")
    return StageResult(stage="intake", ok=True, ms=ms(), payload=payload, warnings=warnings), img


def run(ctx) -> StageResult:
    """Stage entry point. `ctx` needs `.raw_bytes` and may carry `.modality_hint`."""
    raw = getattr(ctx, "raw_bytes", None)
    if raw is None:
        path = getattr(ctx, "path", None)
        if path is None:
            return StageResult(stage="intake", ok=False, ms=0, payload={"error": "no input"}, warnings=["no input"])
        raw = Path(path).read_bytes()
    hint = getattr(ctx, "modality_hint", None)
    router = getattr(ctx, "router", None)
    if router is None and hint is None:
        from medproof.intake.router import get_default_router

        router = get_default_router()
    result, _ = run_bytes(raw, hint, router)
    return result
