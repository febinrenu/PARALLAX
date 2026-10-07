"""The lexicon must cover every label P4's core/vocab.py defines, so no label can be typed outside a slot."""

from __future__ import annotations

import pytest

from medproof.core.vocab import VOCAB
from medproof.report import lexicon

ALL_LABELS = sorted({label for labels in VOCAB.values() for label in labels})


@pytest.mark.parametrize("label", ALL_LABELS)
def test_every_core_vocab_label_is_a_blocked_term(label):
    spoken = label.lower().replace("_", " ")
    assert spoken in lexicon.VOCAB_TERMS
    assert lexicon.VOCAB_RE.search(f"doctor, consider {spoken} here")


def test_labels_added_to_the_core_vocab_are_picked_up_without_editing_the_lexicon(monkeypatch):
    import importlib

    from medproof.core import vocab

    monkeypatch.setitem(vocab.VOCAB, "cxr", (*vocab.VOCAB["cxr"], "Brand_New_Label"))
    reloaded = importlib.reload(lexicon)
    try:
        assert "brand new label" in reloaded.VOCAB_TERMS
    finally:
        monkeypatch.undo()
        importlib.reload(lexicon)
