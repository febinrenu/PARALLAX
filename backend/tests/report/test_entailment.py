from __future__ import annotations

import json

import pytest

from medproof.llm.errors import LLMUnavailable
from medproof.report.drafts import template_drafts, to_claim
from medproof.report.entailment import ClaimVerdict, Verdicts, evidence_for, judge
from medproof.report.firewall import run
from tests.report.conftest import claim


class FakePool:
    def __init__(self, decide=lambda cid: True, exc=None):
        self.decide, self.exc, self.calls = decide, exc, []

    def chat(self, service, prompt_name, user, schema, **kw):
        self.calls.append((service, prompt_name, json.loads(user)))
        if self.exc:
            raise self.exc
        items = json.loads(user)["claims"]

        class R:
            data = Verdicts(verdicts=[ClaimVerdict(claim_id=i["claim_id"], entailed=self.decide(i["claim_id"]), reason="no support for the stated region")
                                       for i in items])

        return R()


def rendered_claims(view):
    out, _ = run([to_claim(d, i, view) for i, d in enumerate(template_drafts(view).drafts, 1)], view)
    return out


def test_evidence_json_has_what_a_judge_needs_and_nothing_hidden(view):
    c = rendered_claims(view)[0]  # f1: supported_by_note
    ev = evidence_for(c, view)
    f = ev["findings"][0]
    assert f["id"] == "f1" and f["label"] == "Consolidation" and f["tier"] == "high" and f["status"] == "verified"
    assert f["region"] == "right lower zone" and f["second_reader"] == "unavailable"
    assert {"id": "te_1", "quote": "fever", "polarity": "supports"} in ev["note_evidence"]
    assert any(i["id"] == "ie_1" and i["faithful"] is True for i in ev["image_evidence"])
    assert "prob" not in json.dumps(ev).lower()  # the sentence never states numbers, so none are offered


def test_second_reader_state_is_spelled_out_not_left_as_null(view):
    from medproof.core.schemas import SecondRead

    f = view.findings["f1"].model_copy(update={"second_read": SecondRead(model="m", label="x", agrees=False, box_iou=None, raw_ref="r")})
    g = view.findings["f9"].model_copy(update={"second_read": SecondRead(model="m", label=None, agrees=None, box_iou=None, raw_ref="r")})
    v2 = type(view).build([f, g], {"n_1": "x"}, {})
    ev = evidence_for(claim("Doctor, consider {f1.label} and {f9.label}.", ["ie_1", "ie_9"]), v2)
    assert [x["second_reader"] for x in ev["findings"]] == ["disagrees", "inconclusive"]


def test_notes_context_counts_what_the_notes_contain(view):
    ev = evidence_for(claim("Doctor, consider {f1.label}.", ["ie_1"]), view)
    ctx = ev["notes_context"]
    assert ctx["symptoms"] >= 1 and ctx["history_or_device_or_medication"] == 0 and ctx["total_facts"] >= ctx["symptoms"]


def test_findings_named_in_slots_are_included_even_if_their_evidence_is_cited_elsewhere(view):
    ev = evidence_for(claim("Doctor, consider {f1.label} and {f9.label}.", ["ie_1", "ie_9"]), view)
    assert [f["id"] for f in ev["findings"]] == ["f1", "f9"]


def test_entailed_claims_are_marked_and_kept(view):
    pool = FakePool()
    out, stage = judge(rendered_claims(view), view, pool)
    assert all(c.entailed is True and c.blocked_reason is None and c.rendered for c in out)
    assert stage.stage == "entailment" and stage.ok and stage.payload["entailed"] == len(out)
    assert pool.calls[0][0] == "judge" and pool.calls[0][1] == "judge"


def test_non_entailed_claims_are_flagged_with_reason_but_the_sentence_is_kept_for_the_audit_view(view):
    claims = rendered_claims(view)
    bad = claims[1].claim_id
    out, stage = judge(claims, view, FakePool(decide=lambda cid: cid != bad))
    hit = next(c for c in out if c.claim_id == bad)
    assert hit.entailed is False and hit.blocked_reason.startswith("E1_not_entailed") and "region" in hit.blocked_reason
    assert hit.rendered == next(c for c in claims if c.claim_id == bad).rendered  # shown struck through
    assert stage.payload["not_entailed"] == 1


def test_claims_already_blocked_by_the_firewall_are_not_sent_to_the_judge(view):
    blocked, _ = run([claim("Doctor, the patient has {f1.label}.", ["ie_1"], "bad")], view)
    pool = FakePool()
    out, _ = judge([*blocked, *rendered_claims(view)[:1]], view, pool)
    sent = [i["claim_id"] for i in pool.calls[0][2]["claims"]]
    assert "bad" not in sent and out[0].entailed is None and out[0].blocked_reason.startswith("R")


def test_judge_outage_leaves_claims_unchecked_and_says_so(view):
    out, stage = judge(rendered_claims(view), view, FakePool(exc=LLMUnavailable("down")))
    assert all(c.entailed is None and c.rendered and c.blocked_reason is None for c in out)
    assert any("not independently checked" in w for w in stage.warnings) and stage.ok is False


def test_no_pool_is_the_same_as_an_outage(view):
    out, stage = judge(rendered_claims(view), view, None)
    assert all(c.entailed is None for c in out) and stage.warnings


def test_missing_verdicts_stay_unchecked_and_unknown_ids_are_ignored(view):
    class Partial(FakePool):
        def chat(self, service, prompt_name, user, schema, **kw):
            first = json.loads(user)["claims"][0]["claim_id"]

            class R:
                data = Verdicts(verdicts=[ClaimVerdict(claim_id=first, entailed=True), ClaimVerdict(claim_id="ghost", entailed=False)])

            return R()

    out, _ = judge(rendered_claims(view), view, Partial())
    assert out[0].entailed is True and all(c.entailed is None for c in out[1:])


def test_claims_are_judged_in_batches_of_six(view):
    many = []
    base = rendered_claims(view)[0]
    for i in range(14):
        many.append(base.model_copy(update={"claim_id": f"c{i}"}))
    pool = FakePool()
    out, _ = judge(many, view, pool)
    assert [len(c[2]["claims"]) for c in pool.calls] == [6, 6, 2] and all(c.entailed for c in out)


def test_inputs_are_not_mutated(view):
    claims = rendered_claims(view)
    judge(claims, view, FakePool(decide=lambda cid: False))
    assert all(c.entailed is None and c.blocked_reason is None for c in claims)


def test_judge_sees_the_sentence_text_not_the_template(view):
    pool = FakePool()
    claims = rendered_claims(view)
    judge(claims, view, pool)
    sent = pool.calls[0][2]["claims"][0]
    assert sent["sentence"] == claims[0].rendered and "{" not in sent["sentence"]


@pytest.mark.parametrize("n", [0])
def test_empty_input_makes_no_call(view, n):
    pool = FakePool()
    out, stage = judge([], view, pool)
    assert out == [] and pool.calls == [] and stage.ok
