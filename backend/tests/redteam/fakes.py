"""Offline stand-ins and builders shared by the red-team cases. Nothing here touches the network."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from medproof.context.contradictions import Demographics
from medproof.context.extract import ExtractedFacts, RawFact
from medproof.core.schemas import Finding, ImageEvidence
from medproof.llm.errors import LLMUnavailable
from medproof.llm.groq_pool import ScoreResult
from medproof.readers.generalist import GeneralistRead
from medproof.reasoning import ReasoningResult, run_reasoning
from medproof.report.drafts import ClaimDraft, ClaimDrafts
from medproof.report.entailment import ClaimVerdict, Verdicts
from medproof.verify.report_labels import labels_from_report

ISSUED = datetime(2026, 10, 7, tzinfo=UTC)
BANNED = ("the patient has", "definitely", "diagnosis is", "ignore previous", "no findings")


def finding(fid: str = "f1", label: str = "Consolidation", region: str | None = "right lower zone", tier: str = "high",
            faithful: bool | None = True, status: str = "uncertain") -> Finding:
    ev = ImageEvidence(evidence_id=f"ie_{fid[1:]}", kind="bbox", bbox_xyxy=(1.0, 1.0, 9.0, 9.0), source_model="xrv@abcd1234",
                       method="gradcam++", region_name=region, faithful=faithful)
    return Finding(finding_id=fid, modality="cxr", label=label, prob_raw=0.8, prob_calibrated=0.8, conformal_set=[], tier=tier,  # type: ignore[arg-type]
                   status=status, image_evidence=[ev])  # type: ignore[arg-type]


class FakeLLM:
    """Answers every prompt by name. `extract_quotes` are what the 'model' returns as fact quotes, whether or not
    they are in the note, so a quote lifted from an injected sentence tests what the code does with it."""

    def __init__(self, extract_quotes: tuple[str, ...] = ("fever", "Right"), entail: bool = True, fail: tuple[str, ...] = ()) -> None:
        self.extract_quotes, self.entail, self.fail = extract_quotes, entail, set(fail)

    def score(self, service: str, model: str, text: str) -> ScoreResult:
        if "score" in self.fail:
            raise LLMUnavailable("prompt guard down")
        return ScoreResult(0.0, "live", model)

    def chat(self, service: str, prompt_name: str, user: str, schema, **kw):
        if prompt_name in self.fail:
            raise LLMUnavailable(f"{prompt_name} down")
        if prompt_name == "guard":
            data = schema(injection=False)
        elif prompt_name == "extract":
            data = ExtractedFacts(facts=[RawFact(type="symptom", value=q, polarity="present", quote=q) for q in self.extract_quotes])
        elif prompt_name == "report":
            data = ClaimDrafts(drafts=[ClaimDraft(frame="consider", finding="f1")])
        elif prompt_name == "judge":
            ids = [c["claim_id"] for c in json.loads(user)["claims"]]
            data = Verdicts(verdicts=[ClaimVerdict(claim_id=i, entailed=self.entail, reason="checked") for i in ids])
        else:
            raise AssertionError(prompt_name)

        class Reply:
            pass

        r = Reply()
        r.data, r.source, r.ok, r.warning = data, "live", True, None  # type: ignore[attr-defined]
        return r


def second_read(text: str = "Consolidation in the right lower zone.") -> GeneralistRead:
    p = labels_from_report(text, "cxr")
    return GeneralistRead(ok=True, modality="cxr", source="live", model="medgemma", findings_text=text, labels=p.labels,
                          says_normal=p.says_normal, raw_ref="medgemma:redteam")


def run_study(notes: dict[str, str], pool: object | None = None, demo: Demographics | None = None, read: GeneralistRead | None = None,
              findings: list[Finding] | None = None, **kw: object) -> ReasoningResult:
    pool = pool if pool is not None else FakeLLM()
    args: dict[str, object] = {"notes": notes, "demographics": demo or Demographics(63, "F"), "generalist_read": read, "text_pool": pool,
                                   "report_pool": pool, "judge_pool": pool, "study_id": "rt", "issued": ISSUED, "ledger_head": "ab" * 32}
    args.update(kw)
    return run_reasoning(findings if findings is not None else [finding()], **args)  # type: ignore[arg-type]
