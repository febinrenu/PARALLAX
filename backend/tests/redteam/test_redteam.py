from __future__ import annotations

import pytest

from tests.redteam.cases import CASES


def test_the_suite_has_at_least_fifty_cases_across_every_category():
    assert len(CASES) >= 50
    assert {c.category for c in CASES} == {"notes_robustness", "contradictions", "injection", "firewall", "degradation", "policy", "stages"}
    assert len({c.id for c in CASES}) == len(CASES)


@pytest.mark.parametrize("c", CASES, ids=[f"{c.category}:{c.id}" for c in CASES])
def test_case(c):
    passed, detail = c.run()
    assert passed, f"{c.id}: expected {c.expected}; got {detail}"
