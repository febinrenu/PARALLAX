"""The status rule (plan.md section 4), enforced in one function and tested exhaustively.

Precedence matters and is checked top to bottom: a finding with no valid evidence is
`rejected` even if a second reader happens to agree; a `discordant` finding stays discordant
even if its own evidence would otherwise look verified. Only a finding that clears every bar
is `verified`.
"""

from __future__ import annotations

from medproof.core.schemas import Finding, Status

FLIP_RATE_LIMIT = 0.25
CONFORMAL_SET_LIMIT = 2
BOX_IOU_DISCORDANT_BELOW = 0.1


def _has_valid_evidence(finding: Finding) -> bool:
    if any(ie.faithful is True for ie in finding.image_evidence):
        return True
    return any(te.polarity == "supports" for te in finding.text_evidence)


def _is_discordant(finding: Finding) -> bool:
    second = finding.second_read
    if second is None:
        return False
    if second.agrees is False:
        return True
    return second.box_iou is not None and second.box_iou < BOX_IOU_DISCORDANT_BELOW


def _is_uncertain(finding: Finding) -> bool:
    # P3-2: when the finding has an image region, only a region that passed the deletion test can make
    # it verified; supporting note text alone lifts it to uncertain at most.
    if finding.image_evidence and not any(ie.faithful is True for ie in finding.image_evidence):
        return True
    if finding.stability is not None and finding.stability.flip_rate > FLIP_RATE_LIMIT:
        return True
    if "ood" in finding.flags:
        return True
    if len(finding.conformal_set) > CONFORMAL_SET_LIMIT:
        return True
    # "abstain" (P2's calibrated tier below "low") is weaker than "low", so it is uncertain too.
    return finding.tier in ("low", "abstain")


def compute_status(finding: Finding) -> Status:
    """Pure function: `Finding` in, `Status` out. No side effects, no I/O."""
    if not _has_valid_evidence(finding):
        return "rejected"
    if _is_discordant(finding):
        return "discordant"
    if _is_uncertain(finding):
        return "uncertain"
    return "verified"
