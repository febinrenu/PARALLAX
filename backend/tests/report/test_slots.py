import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from medproof.report.slots import Slot, TemplateSyntaxError, parse_template

SLOT_STRINGS = ["{f1.label}", "{f1.region}", "{te_1.quote}", "{ie_1.region}", "{f2.tier}"]
chunks = st.one_of(st.sampled_from(SLOT_STRINGS), st.text(max_size=12))


@given(st.lists(chunks, min_size=1, max_size=8).map("".join))
@settings(max_examples=400, deadline=None)
def test_accepted_templates_have_no_digits_or_braces_in_literal_text(template):
    try:
        parsed = parse_template(template)
    except TemplateSyntaxError:
        return
    for part in parsed.parts:
        if isinstance(part, str):
            assert not any(ch.isdigit() for ch in part)
            assert "{" not in part and "}" not in part
            assert part.isascii()
    assert any(isinstance(p, Slot) and p.kind == "f" for p in parsed.parts)


def test_digit_in_literal_text_is_rejected():
    with pytest.raises(TemplateSyntaxError):
        parse_template("Doctor, consider {f1.label} in 2 zones.")


def test_slot_digits_are_part_of_the_ref_only():
    p = parse_template("Doctor, consider {f12.label}.")
    assert p.slots == [Slot("f12", "label")]


def test_report_vocabulary_matches_the_calibration_tiers():
    # P2's calibrator emits these tiers; the renderer and the confidence lexicon must know every one.
    from medproof.calibrate.abstain import TIERS
    from medproof.report import lexicon

    assert set(TIERS) == set(lexicon.TIER_PHRASE)
