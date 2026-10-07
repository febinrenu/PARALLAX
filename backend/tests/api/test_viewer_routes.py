"""Routes the workstation viewer needs (P4.6/P4.7): display image, full-depth pixels, per-study
artifacts and ledger, and the guarantee that no server filesystem path reaches a client."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from tests.api.test_studies import _client, _poll_until_done, _stub_stages, _upload
from tests.conftest import png_bytes

torch = pytest.importorskip("torch")


def _done_study(tmp_path, client=None):
    client = client or _client(tmp_path, stages=_stub_stages())
    study_id = _upload(client).json()["study_id"]
    return client, study_id, _poll_until_done(client, study_id)


def test_image_returns_the_original_size_png(tmp_path):
    client, study_id, _ = _done_study(tmp_path)
    resp = client.get(f"/studies/{study_id}/image")
    assert resp.status_code == 200 and resp.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(resp.content)).size == (500, 300)  # tests.readers.test_cxr HW


def test_pixels_are_float16_luminance_with_dimension_headers(tmp_path):
    client, study_id, _ = _done_study(tmp_path)
    resp = client.get(f"/studies/{study_id}/pixels")
    assert resp.status_code == 200
    w, h = int(resp.headers["x-width"]), int(resp.headers["x-height"])
    assert (w, h) == (500, 300)
    pixels = np.frombuffer(resp.content, dtype="<f2")
    assert pixels.size == w * h
    assert 0.0 <= float(pixels.min()) and float(pixels.max()) <= 1.0


def test_pixels_downsample_the_long_edge_to_2048(tmp_path):
    client = _client(tmp_path)
    wide = (np.random.default_rng(0).random((200, 3000)) * 255).astype(np.uint8)
    files = {"image": ("wide.png", io.BytesIO(png_bytes(wide)), "image/png")}
    study_id = client.post("/studies", files=files).json()["study_id"]
    resp = client.get(f"/studies/{study_id}/pixels")
    assert int(resp.headers["x-width"]) == 2048
    assert int(resp.headers["x-original-width"]) == 3000
    assert np.frombuffer(resp.content, dtype="<f2").size == 2048 * int(resp.headers["x-height"])


def test_heatmap_refs_are_api_urls_and_resolve(tmp_path):
    client, study_id, result = _done_study(tmp_path)
    refs = [ev["heatmap_ref"] for f in result["findings"] for ev in f["image_evidence"] if ev["heatmap_ref"]]
    assert refs, "stub reader should produce heatmaps"
    for ref in refs:
        assert ref.startswith(f"/studies/{study_id}/artifacts/")
        assert client.get(ref).status_code == 200


def test_no_server_filesystem_path_reaches_any_response(tmp_path):
    client, study_id, _ = _done_study(tmp_path)
    leak = str(tmp_path).replace("\\", "/")
    bodies = [
        client.get(f"/studies/{study_id}").text,
        client.get(f"/studies/{study_id}/ledger").text,
    ]
    with client.stream("GET", f"/studies/{study_id}/events") as resp:
        bodies.append("".join(resp.iter_lines()))
    for body in bodies:
        normalized = body.replace("\\\\", "/").replace("\\", "/")
        assert leak not in normalized


@pytest.mark.parametrize("name", ["..%2F..%2Fsecret.png", "heatmap.txt", "a b.png", "%2e%2e.png"])
def test_artifact_names_outside_the_whitelist_are_404(tmp_path, name):
    client, study_id, _ = _done_study(tmp_path)
    assert client.get(f"/studies/{study_id}/artifacts/{name}").status_code == 404


def test_missing_artifact_is_404(tmp_path):
    client, study_id, _ = _done_study(tmp_path)
    assert client.get(f"/studies/{study_id}/artifacts/nope.png").status_code == 404


def test_study_ledger_only_returns_that_studys_entries(tmp_path):
    client, first, _ = _done_study(tmp_path)
    _, second, _ = _done_study(tmp_path, client=client)
    entries = client.get(f"/studies/{first}/ledger").json()["entries"]
    assert entries and all(e["study_id"] == first for e in entries)
    assert second not in {e["study_id"] for e in entries}


def test_events_are_not_gzip_encoded(tmp_path):
    client, study_id, _ = _done_study(tmp_path)
    with client.stream("GET", f"/studies/{study_id}/events") as resp:
        assert "gzip" not in resp.headers.get("content-encoding", "")


def test_artifacts_resolve_for_a_repeat_upload_served_from_the_stage_cache(tmp_path):
    """Second upload of the same image hits the stage cache; its heatmap refs must still resolve."""
    client, _, _ = _done_study(tmp_path)
    _, second, result = _done_study(tmp_path, client=client)
    refs = [ev["heatmap_ref"] for f in result["findings"] for ev in f["image_evidence"] if ev["heatmap_ref"]]
    assert refs
    for ref in refs:
        assert ref.startswith(f"/studies/{second}/artifacts/")
        assert client.get(ref).status_code == 200
