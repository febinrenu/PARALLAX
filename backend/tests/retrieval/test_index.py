from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("faiss")

from medproof.core.schemas import Precedent  # noqa: E402
from medproof.retrieval.index import PrecedentIndex, precision_at_k, to_precedents  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype="float32")
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make_index(n_per_class=6, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    centers = {"nv": np.eye(dim)[0], "mel": np.eye(dim)[1], "bkl": np.eye(dim)[2]}
    ids, labels, vecs, groups = [], [], [], []
    for lab, c in centers.items():
        for i in range(n_per_class):
            ids.append(f"{lab}_{i}")
            labels.append(lab)
            groups.append(f"g_{lab}_{i // 2}")  # two images per lesion group
            vecs.append(c + 0.15 * rng.standard_normal(dim))
    return PrecedentIndex.build("ham10000", "fake-embedder", ids, labels, unit(np.array(vecs)), groups=groups), centers


def test_nearest_neighbours_come_from_the_query_s_own_class():
    idx, centers = make_index()
    hits = idx.search(unit(centers["mel"] + 0.05), k=5)
    assert len(hits) == 5 and all(h.label == "mel" for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(-1.0001 <= h.score <= 1.0001 for h in hits)


def test_the_query_image_and_its_lesion_group_are_never_returned():
    idx, _ = make_index()
    qvec = idx.vectors[idx.ids.index("mel_0")]
    hits = idx.search(qvec, k=5, exclude_ids=["mel_0"], exclude_groups=["g_mel_0"])
    assert "mel_0" not in [h.id for h in hits] and "mel_1" not in [h.id for h in hits]  # mel_1 shares the group


def test_k_larger_than_the_index_returns_everything_and_empty_index_returns_nothing():
    idx, centers = make_index(n_per_class=2)
    assert len(idx.search(unit(centers["nv"]), k=50)) == 6
    empty = PrecedentIndex.build("ham10000", "fake", [], [], np.zeros((0, 8), dtype="float32"))
    assert empty.search(unit(np.ones(8)), k=5) == []


def test_save_and_load_round_trip_gives_identical_results(tmp_path):
    idx, centers = make_index()
    idx.save(tmp_path / "ham")
    again = PrecedentIndex.load(tmp_path / "ham")
    q = unit(centers["bkl"] + 0.1)
    assert [(h.id, round(h.score, 5)) for h in again.search(q, 5)] == [(h.id, round(h.score, 5)) for h in idx.search(q, 5)]
    assert again.embedder == "fake-embedder" and again.dataset == "ham10000"


def test_loading_with_a_different_embedder_is_refused(tmp_path):
    idx, _ = make_index()
    idx.save(tmp_path / "ham")
    with pytest.raises(ValueError, match="embedder"):
        PrecedentIndex.load(tmp_path / "ham", expect_embedder="medsiglip-448")


def test_building_rejects_mismatched_inputs():
    with pytest.raises(ValueError):
        PrecedentIndex.build("d", "e", ["a", "b"], ["x"], np.zeros((2, 4), dtype="float32"))
    with pytest.raises(ValueError):
        PrecedentIndex.build("d", "e", ["a"], ["x"], np.zeros((1, 4), dtype="float32"), groups=["g1", "g2"])


def test_hits_become_contract_precedents_with_thumbnails():
    idx, centers = make_index()
    hits = idx.search(unit(centers["nv"]), k=3)
    precedents = to_precedents(hits, "ham10000", lambda h: f"thumbs/ham10000/{h.id}.jpg")
    assert all(isinstance(p, Precedent) for p in precedents)
    assert precedents[0].case_id == hits[0].id and precedents[0].label == "nv"
    assert precedents[0].thumb_ref == f"thumbs/ham10000/{hits[0].id}.jpg" and 0.0 <= precedents[0].similarity <= 1.0


def test_similarity_in_a_precedent_is_clamped_to_the_unit_interval():
    class H:
        id, label, score = "x", "nv", -0.2

    assert to_precedents([H()], "d", lambda h: "t")[0].similarity == 0.0  # type: ignore[list-item]


def test_precision_at_k_matches_a_hand_count():
    assert precision_at_k([["a", "a", "b", "a", "c"]], ["a"]) == pytest.approx(3 / 5)
    assert precision_at_k([["a", "a"], ["b", "x"]], ["a", "b"]) == pytest.approx((1.0 + 0.5) / 2)
    assert precision_at_k([[]], ["a"]) == 0.0
