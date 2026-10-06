"""Scores the entailment judge on 60 labelled sentences per variant (plan 9.5: accuracy >= 0.9).

30 sentences are true to their evidence (valid frames rendered by the real renderer, plus truthful
second-reader, flag and label-set clauses). 30 are the same sentences with one known defect added by
string surgery: wrong label or region, inflated tier, invented second-reader agreement, an invented
quote, an inflated status, or an extra clinical claim. Labels follow from how each case was built.

Two variants use different findings, regions, quotes and note text. Variant 0 was used to find two
gaps in what the judge is shown (first-pass result kept in reports/p3_entailment_first_pass.json);
variant 1 was written afterwards and is the unbiased measurement of the fixed judge.

    python -m ml.eval_p3.eval_entailment --variant 1     # live Groq on the first run, cached afterwards
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from medproof.core.schemas import Claim, Finding, ImageEvidence, SecondRead, TextEvidence  # noqa: E402
from medproof.llm.config import LLMConfig  # noqa: E402
from medproof.llm.groq_pool import GroqPool  # noqa: E402
from medproof.report.entailment import judge  # noqa: E402
from medproof.report.firewall import check  # noqa: E402
from medproof.report.study_view import StudyView  # noqa: E402

from ml.eval_p3.eval_context import load_env  # noqa: E402

NOTE_ID = "n1"
# (id, modality, label, tier, status, region, second reader, flags, conformal set, supporting quote, faithful)
VARIANTS: dict[int, dict[str, Any]] = {
    0: {
        "note": "Fever and dyspnea for 3 days, chest pain on breathing, weight loss, headache, pain after a fall.",
        "fake_quotes": ["hemoptysis", "orthopnea", "syncope", "numbness"],
        "wrong_labels": {1: "Pneumothorax", 2: "Mass", 3: "Consolidation", 5: "pneumonia", 7: "meningioma"},
        "findings": [
            ("f1", "cxr", "Consolidation", "high", "verified", "right lower zone", True, [], ["Consolidation", "Pneumonia"], "Fever", True),
            ("f2", "cxr", "Effusion", "moderate", "uncertain", "left base", False, ["ood"], [], "dyspnea", True),
            ("f3", "cxr", "Pneumothorax", "low", "uncertain", "right upper zone", "none", [], [], "chest pain", None),
            ("f4", "skin_dermoscopy", "mel", "high", "verified", None, "none", [], [], None, True),
            ("f5", "bone_xray", "fracture", "high", "verified", "left wrist", "none", [], [], "pain after a fall", True),
            ("f6", "cxr", "Nodule", "abstain", "uncertain", "left upper zone", "none", [], [], "weight loss", None),
            ("f7", "brain_mri", "glioma", "moderate", "verified", "right frontal", True, [], [], "headache", True),
            ("f8", "cxr", "Cardiomegaly", "moderate", "discordant", "cardiac", False, ["laterality_conflict"], [], None, True),
        ],
    },
    1: {
        "note": "Productive cough with night sweats, swelling of the ankle after exertion, blurred vision, bruising on the thigh, itching of the lesion.",
        "fake_quotes": ["jaundice", "palpitations", "seizure", "hearing loss"],
        "wrong_labels": {1: "Effusion", 2: "Nodule", 3: "Pneumothorax", 5: "glioma", 7: "bcc"},
        "findings": [
            ("f1", "cxr", "Pneumonia", "high", "verified", "left lower zone", True, [], ["Pneumonia", "Consolidation"], "Productive cough", True),
            ("f2", "cxr", "Atelectasis", "moderate", "uncertain", "right base", False, ["low_quality"], [], "night sweats", True),
            ("f3", "cxr", "Edema", "low", "uncertain", "left upper zone", "none", [], [], "swelling of the ankle", None),
            ("f4", "skin_dermoscopy", "bcc", "high", "verified", None, "none", [], [], None, True),
            ("f5", "bone_xray", "fracture", "high", "verified", "right femur", "none", [], [], "bruising on the thigh", True),
            ("f6", "cxr", "Mass", "abstain", "uncertain", "right upper zone", "none", [], [], "night sweats", None),
            ("f7", "brain_mri", "meningioma", "moderate", "verified", "left parietal", True, [], [], "blurred vision", True),
            ("f8", "cxr", "Effusion", "moderate", "discordant", "right base", False, ["partial_view"], [], None, True),
        ],
    },
}


def build_view(variant: int) -> StudyView:
    spec = VARIANTS[variant]
    note = spec["note"]
    fs: list[Finding] = []
    for fid, modality, label, tier, status, region, agrees, flags, conformal, quote, faithful in spec["findings"]:
        n = fid[1:]
        ie = ImageEvidence(evidence_id=f"ie_{n}", kind="bbox", bbox_xyxy=(1.0, 1.0, 9.0, 9.0), source_model="model@abcd1234",
                           method="gradcam++", region_name=region, faithful=faithful)
        tes = []
        if quote:
            i = note.index(quote)
            tes = [TextEvidence(evidence_id=f"te_{n}0", note_id=NOTE_ID, span=(i, i + len(quote)), quote=quote, fact_type="symptom", polarity="supports")]
        sr = None if agrees == "none" else SecondRead(model="medgemma-1.5-4b-it", label=label, agrees=agrees, box_iou=None, raw_ref="r")
        fs.append(Finding(finding_id=fid, modality=modality, label=label, prob_raw=0.8, prob_calibrated=0.8, conformal_set=conformal,
                          tier=tier, status=status, image_evidence=[ie], text_evidence=tes, second_read=sr, flags=flags))
    return StudyView.build(fs, {NOTE_ID: note}, {})


def claim(view: StudyView, cid: str, template: str, ids: list[str]) -> Claim:
    c = Claim(claim_id=cid, template=template, evidence_ids=ids)
    v = check(c, view)
    if v.rendered is None:
        raise AssertionError(f"positive case not valid for the firewall: {template} -> {v.reasons}")
    return c.model_copy(update={"rendered": v.rendered})


def positives(view: StudyView) -> list[tuple[str, Claim]]:
    out: list[tuple[str, Claim]] = []
    n = 0

    def add(kind: str, template: str, ids: list[str]) -> None:
        nonlocal n
        n += 1
        out.append((kind, claim(view, f"p{n}", template, ids)))

    for i in (1, 2, 3, 4, 5, 7):  # f6 (abstain) and f8 (discordant) cannot use the plain 'consider' frame
        add("consider", f"Doctor, consider {{f{i}.label}} in the {{f{i}.region}} ({{f{i}.tier}}).", [f"ie_{i}"])
    for i in (1, 2, 3, 5, 7):
        add("supported_by_note", f"Doctor, consider {{f{i}.label}} ({{f{i}.tier}}); the note states {{te_{i}0.quote}}.", [f"ie_{i}", f"te_{i}0"])
    for i in (2, 8):
        add("discordant", f"Doctor, the two readers disagree on {{f{i}.label}}; please review the image directly.", [f"ie_{i}"])
    add("abstain", "Doctor, there is no confident reading for {f6.label}; clinical correlation is needed.", ["ie_6"])
    for i in (1, 2, 3, 4, 5, 7):
        add("context_missing", f"Doctor, the notes do not mention history relevant to {{f{i}.label}}; please add context.", [f"ie_{i}"])
    for i in (1, 7, 2, 3, 4, 5):
        add("second_read_clause", f"Doctor, consider {{f{i}.label}} ({{f{i}.tier}}); {{f{i}.second_read}}.", [f"ie_{i}"])
    add("flag_clause", "Doctor, the two readers disagree on {f8.label} with {f8.flag_phrase}; please review the image directly.", ["ie_8"])
    add("flag_clause", "Doctor, consider {f2.label} ({f2.tier}) with {f2.flag_phrase}.", ["ie_2"])
    add("flag_clause", "Doctor, consider {f3.label} ({f3.tier}) with {f3.flag_phrase}.", ["ie_3"])
    add("label_set", "Doctor, consider {f1.label}; the label set is {f1.conformal_set}.", ["ie_1"])
    return out


def swap_side(text: str) -> str:
    for a, b in (("left", "right"), ("right", "left")):
        if a in text:
            return text.replace(a, b, 1)
    return text


def negatives(view: StudyView, pos: dict[str, Claim], variant: int) -> list[tuple[str, Claim]]:
    spec = VARIANTS[variant]

    def consider(i: int) -> Claim:
        return next(c for c in pos.values() if c.template.startswith(f"Doctor, consider {{f{i}.label}} in the"))

    def discordant(i: int) -> Claim:
        return next(c for c in pos.values() if c.template.startswith(f"Doctor, the two readers disagree on {{f{i}.label}}"))

    def note(i: int) -> Claim:
        return next(c for c in pos.values() if f"{{te_{i}0.quote}}" in c.template)

    def mut(kind: str, base: Claim, new: str) -> tuple[str, Claim]:
        assert new != base.rendered, (kind, new)
        return kind, base.model_copy(update={"rendered": new})

    out: list[tuple[str, Claim]] = []
    for i, wrong in spec["wrong_labels"].items():
        b = consider(i)
        out.append(mut("wrong_label", b, b.rendered.replace(view.findings[f"f{i}"].label, wrong)))
    for i in (1, 2, 3, 5, 7):
        b = consider(i)
        region = view.findings[f"f{i}"].image_evidence[0].region_name or ""
        out.append(mut("wrong_region", b, b.rendered.replace(region, swap_side(region))))
    swaps = {1: ("high confidence", "low confidence"), 3: ("low confidence", "high confidence"), 2: ("moderate confidence", "high confidence"),
             7: ("moderate confidence", "low confidence"), 5: ("high confidence", "low confidence")}
    for i, (old, new) in swaps.items():
        b = consider(i)
        out.append(mut("wrong_tier", b, b.rendered.replace(old, new)))
    for i in (2, 8, 3, 4, 5):
        b = discordant(i) if i == 8 else consider(i)
        out.append(mut("invented_second_read", b, b.rendered.rstrip(".") + "; the second reader agrees."))
    for i, fake in zip((1, 2, 3, 5), spec["fake_quotes"], strict=True):
        b = note(i)
        old = view.text_evidence[f"te_{i}0"].quote
        out.append(mut("invented_quote", b, b.rendered.replace(f'"{old}"', f'"{fake}"')))
    for i in (2, 3, 8):
        b = discordant(i) if i == 8 else consider(i)
        out.append(mut("inflated_status", b, b.rendered.rstrip(".") + "; this finding is verified."))
    for i, extra in ((1, "; this is consistent with tuberculosis."), (5, "; urgent surgical referral is recommended."),
                     (7, "; this represents a new finding compared with the prior film.")):
        b = consider(i)
        out.append(mut("extra_claim", b, b.rendered.rstrip(".") + extra))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=int, choices=sorted(VARIANTS), default=1)
    args = ap.parse_args(argv)
    load_env()
    view = build_view(args.variant)
    pos = positives(view)
    posmap = {f"{k}{n}": c for n, (k, c) in enumerate(pos)}
    neg = negatives(view, posmap, args.variant)
    cases = [(True, k, c) for k, c in pos] + [(False, k, c) for k, c in neg]
    assert len(pos) == 30 and len(neg) == 30, (len(pos), len(neg))
    random.Random(20261007).shuffle(cases)  # every batch of six mixes true and defective sentences
    pool = GroqPool(LLMConfig(cache_dir=ROOT / ".cache" / "llm", deadline_s=300.0, max_attempts=6))
    unique = [c.model_copy(update={"claim_id": f"k{i}"}) for i, (_, _, c) in enumerate(cases)]
    judged, stage = judge(unique, view, pool)
    tp = tn = fp = fn = unchecked = 0
    errors: list[dict[str, Any]] = []
    by_kind: dict[str, list[int]] = {}
    for (truth, kind, _), j in zip(cases, judged, strict=True):
        if j.entailed is None:
            unchecked += 1
            continue
        ok = j.entailed == truth
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0] += int(ok)
        by_kind[kind][1] += 1
        tp += int(truth and j.entailed)
        tn += int((not truth) and (not j.entailed))
        fp += int((not truth) and j.entailed)
        fn += int(truth and (not j.entailed))
        if not ok:
            errors.append({"truth": truth, "kind": kind, "sentence": j.rendered, "judge_reason": j.blocked_reason})
    n = tp + tn + fp + fn
    report = {
        "variant": args.variant, "cases": len(cases), "judged": n, "unchecked": unchecked,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "entailed_sentences_kept": f"{tp}/{tp + fn}", "defective_sentences_caught": f"{tn}/{tn + fp}",
        "by_kind": {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())}, "errors": errors,
        "judge_model": pool.config.models["judge"], "stage_warnings": stage.warnings,
    }
    path = ROOT / "reports" / f"p3_entailment_variant{args.variant}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))
    pool.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
