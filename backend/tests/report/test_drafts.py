from __future__ import annotations

import pytest

from medproof.llm.errors import LLMUnavailable
from medproof.report.drafts import ClaimDraft, ClaimDrafts, build_input, draft_claims, template_drafts, to_claim
from medproof.report.firewall import run
from tests.report.conftest import make_view


class FakePool:
    def __init__(self, drafts=None, exc=None):
        self.drafts, self.exc, self.calls = drafts, exc, []

    def chat(self, service, prompt_name, user, schema, *, fallback=None, **kw):
        self.calls.append((service, prompt_name, user))
        if self.exc:
            if fallback is None:
                raise self.exc

            class R:
                data = fallback()
                source = "fallback"
                ok = False
                warning = str(self.exc)

            return R()

        class R:  # noqa: F811
            data = self.drafts
            source = "live"
            ok = True
            warning = None

        return R()


def test_template_drafts_skip_rejected_findings_and_pick_sensible_frames(view):
    drafts = template_drafts(view)
    by_finding = {d.finding: d for d in drafts.drafts}
    assert "f2" not in by_finding  # rejected
    assert by_finding["f4"].frame == "discordant_review"
    assert by_finding["f8"].frame == "abstain_note"
    assert by_finding["f1"].frame == "supported_by_note" and by_finding["f1"].note_ref == "te_1"
    assert by_finding["f9"].frame == "consider"


def test_template_claims_all_pass_the_firewall(view):
    claims = [to_claim(d, i, view) for i, d in enumerate(template_drafts(view).drafts, 1)]
    out, stage = run(claims, view)
    assert len(out) == 5 and stage.payload["blocked"] == 0 and stage.payload["rendered"] == 5
    assert all(c.rendered.startswith("Doctor,") for c in out)


def test_to_claim_builds_a_slot_template_with_the_findings_evidence(view):
    c = to_claim(ClaimDraft(frame="supported_by_note", finding="f1", note_ref="te_1"), 3, view)
    assert c.claim_id == "c3"
    assert c.template == "Doctor, consider {f1.label} ({f1.tier}); the note states {te_1.quote}."
    assert c.evidence_ids == ["ie_1", "te_1"]
    plain = to_claim(ClaimDraft(frame="consider", finding="f1"), 1, view)
    assert plain.template == "Doctor, consider {f1.label} in the {f1.region} ({f1.tier})." and plain.evidence_ids == ["ie_1"]


def test_llm_chooses_frames_but_the_firewall_still_decides(view):
    bad = ClaimDrafts(drafts=[
        ClaimDraft(frame="consider", finding="f8"),  # abstain tier cannot be 'consider'
        ClaimDraft(frame="supported_by_note", finding="f1", note_ref="te_1"),
        ClaimDraft(frame="consider", finding="f2"),  # rejected finding
        ClaimDraft(frame="consider", finding="f99"),  # does not exist
        ClaimDraft(frame="supported_by_note", finding="f1", note_ref="te_99"),  # unknown note ref
    ])
    pool = FakePool(bad)
    claims, stage = draft_claims(view, pool)
    rendered = [c.rendered for c in claims if c.rendered]
    assert len(rendered) == 1 and "the note states" in rendered[0]
    assert stage.stage == "report" and stage.payload["blocked"] == 4
    assert pool.calls[0][0] == "report"


def test_llm_input_has_ids_and_states_but_no_free_text(view):
    text = build_input(view)
    assert "f1" in text and "te_1" in text and "tier" in text
    assert "fever" not in text and "Consolidation" not in text  # quotes and labels never reach the model


def test_offline_fallback_when_groq_is_unavailable(view):
    claims, stage = draft_claims(view, FakePool(exc=LLMUnavailable("down")))
    assert claims and all(c.rendered for c in claims[:6])
    assert any("template" in w for w in stage.warnings)


def test_no_pool_uses_templates_directly(view):
    claims, stage = draft_claims(view, None)
    assert claims and any("template" in w for w in stage.warnings)


def test_draft_schema_only_allows_known_frames():
    with pytest.raises(ValueError):
        ClaimDraft(frame="the patient has", finding="f1")


def test_empty_study_gives_no_claims_and_a_clear_stage():
    claims, stage = draft_claims(make_view().__class__.build([], {}, {}), None)
    assert claims == [] and stage.ok
