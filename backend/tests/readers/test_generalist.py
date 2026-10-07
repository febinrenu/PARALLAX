from __future__ import annotations

import io

import httpx
import numpy as np
from PIL import Image

from medproof.intake.decode import DecodedImage
from medproof.readers.generalist import GeneralistReader, norm_box_to_pixels

REPORT = {
    "ok": True,
    "model": "google/medgemma-1.5-4b-it",
    "quant": "nf4/bf16",
    "labels": [],
    "boxes": [{"label": "Effusion", "xyxy_norm": [100, 200, 500, 800]}],
    "impression": "Small right pleural effusion.",
    "findings_text": "There is a small right pleural effusion. No pneumothorax.",
    "raw": "FINDINGS: ...",
    "error": None,
    "warnings": [],
    "ms": 4200,
}


def decoded(h: int = 200, w: int = 400, sha: str = "a" * 64) -> DecodedImage:
    img = (np.arange(h * w).reshape(h, w) % 255).astype(np.uint8)
    return DecodedImage(display=img, analysis=img.astype(np.float32) / 255.0, sha256=sha, source_format="png")


class Server:
    def __init__(self, status=200, body=None, exc=None):
        self.status, self.body, self.exc, self.requests = status, body or REPORT, exc, []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.exc:
            raise self.exc
        return httpx.Response(self.status, json=self.body)


def reader(tmp_path, server, url="http://medgemma.test"):
    return GeneralistReader(url, cache_dir=tmp_path / "mg", transport=httpx.MockTransport(server), timeout=5)


def test_box_conversion_uses_the_display_grid():
    assert norm_box_to_pixels((100, 200, 500, 800), height=200, width=400) == (40.0, 40.0, 200.0, 160.0)
    assert norm_box_to_pixels((0, 0, 1000, 1000), height=512, width=256) == (0.0, 0.0, 256.0, 512.0)


def test_successful_read_derives_labels_and_pixel_boxes(tmp_path):
    srv = Server()
    r = reader(tmp_path, srv).read(decoded(), "cxr")
    assert r.ok and r.source == "live" and r.model.startswith("google/medgemma")
    assert r.labels_state("Effusion") == "present" and r.labels_state("Pneumothorax") == "absent"
    assert r.boxes == [("Effusion", (40.0, 40.0, 200.0, 160.0))]
    assert r.impression == "Small right pleural effusion."
    req = srv.requests[0]
    assert req.url.path == "/read" and b"filename" in req.content and b"cxr" in req.content
    assert req.content.count(b"\x89PNG") == 1  # the display image went out as a PNG


def test_second_read_hits_the_cache(tmp_path):
    srv = Server()
    rd = reader(tmp_path, srv)
    first = rd.read(decoded(), "cxr")
    again = reader(tmp_path, srv).read(decoded(), "cxr")  # new client, same cache dir
    assert again.source == "cache" and again.labels_state("Effusion") == "present"
    assert len(srv.requests) == 1 and again.raw_ref == first.raw_ref
    other = rd.read(decoded(sha="b" * 64), "cxr")
    assert other.source == "live" and len(srv.requests) == 2


def test_cache_is_keyed_by_modality_and_prompt_version(tmp_path):
    srv = Server()
    rd = reader(tmp_path, srv)
    rd.read(decoded(), "cxr")
    rd.read(decoded(), "bone_xray")
    assert len(srv.requests) == 2
    GeneralistReader("http://medgemma.test", cache_dir=tmp_path / "mg", transport=httpx.MockTransport(srv), prompt_version="v9").read(decoded(), "cxr")
    assert len(srv.requests) == 3


def test_connection_failure_degrades_instead_of_raising(tmp_path):
    srv = Server(exc=httpx.ConnectError("refused"))
    r = reader(tmp_path, srv).read(decoded(), "cxr")
    assert not r.ok and r.source == "unavailable" and r.labels == [] and r.boxes == []
    assert any("second read unavailable" in w for w in r.warnings)


def test_server_error_and_timeout_degrade(tmp_path):
    assert not reader(tmp_path, Server(status=503)).read(decoded(), "cxr").ok
    assert not reader(tmp_path, Server(exc=httpx.ReadTimeout("slow"))).read(decoded(sha="c" * 64), "cxr").ok


def test_failed_service_result_is_not_cached(tmp_path):
    bad = dict(REPORT, ok=False, error="refusal: model refused to read the image", labels=[], boxes=[], impression="", findings_text="")
    srv = Server(body=bad)
    rd = reader(tmp_path, srv)
    r = rd.read(decoded(), "cxr")
    assert not r.ok and any("refusal" in w for w in r.warnings)
    rd.read(decoded(), "cxr")
    assert len(srv.requests) == 2


def test_missing_url_means_unavailable_without_a_request(tmp_path):
    srv = Server()
    r = GeneralistReader(None, cache_dir=tmp_path / "mg", transport=httpx.MockTransport(srv), env={}).read(decoded(), "cxr")
    assert not r.ok and srv.requests == [] and any("MEDGEMMA_URL" in w for w in r.warnings)


def test_color_and_grayscale_displays_both_encode(tmp_path):
    srv = Server()
    rgb = decoded()
    rgb.display = np.stack([rgb.display] * 3, axis=-1)
    assert reader(tmp_path, srv).read(rgb, "skin_dermoscopy").ok
    body = srv.requests[0].content
    start = body.index(b"\x89PNG")
    assert Image.open(io.BytesIO(body[start : body.index(b"IEND", start) + 8])).size == (400, 200)


# ---- shipped reads for the offline demo (no GPU, no tunnel, no cache)

def test_a_shipped_read_answers_without_any_service_or_cache(tmp_path):
    seed = tmp_path / "seed"
    live = reader(tmp_path / "a", Server())
    live.export_seed(decoded(), "cxr", seed)
    files = list(seed.glob("*.json"))
    assert len(files) == 1 and files[0].name == f"{'a' * 64}.cxr.v4.json"
    srv = Server(exc=httpx.ConnectError("no gpu here"))
    offline = GeneralistReader("http://nowhere.test", cache_dir=tmp_path / "b", transport=httpx.MockTransport(srv), seed_dir=seed)
    r = offline.read(decoded(), "cxr")
    assert r.ok and r.source == "seed" and r.labels_state("Effusion") == "present" and srv.requests == []


def test_a_seed_for_another_prompt_version_or_modality_is_not_used(tmp_path):
    seed = tmp_path / "seed"
    reader(tmp_path / "a", Server()).export_seed(decoded(), "cxr", seed)
    srv = Server(exc=httpx.ConnectError("down"))
    for modality, version in (("bone_xray", "v4"), ("cxr", "v9")):
        r = GeneralistReader("http://x.test", cache_dir=tmp_path / f"c{modality}{version}", transport=httpx.MockTransport(srv),
                             seed_dir=seed, prompt_version=version).read(decoded(), modality)
        assert not r.ok and r.source == "unavailable"


def test_export_seed_refuses_to_write_a_failed_read(tmp_path):
    bad = dict(REPORT, ok=False, error="refusal", labels=[], boxes=[], impression="", findings_text="")
    import pytest

    with pytest.raises(ValueError):
        reader(tmp_path / "a", Server(body=bad)).export_seed(decoded(), "cxr", tmp_path / "seed")
    assert not (tmp_path / "seed").exists() or not list((tmp_path / "seed").glob("*.json"))


def test_a_corrupt_seed_file_is_ignored(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / f"{'a' * 64}.cxr.v4.json").write_text("{not json", encoding="utf-8")
    srv = Server()
    r = GeneralistReader("http://x.test", cache_dir=tmp_path / "c", transport=httpx.MockTransport(srv), seed_dir=seed).read(decoded(), "cxr")
    assert r.ok and r.source == "live"


def test_export_folder_writes_one_file_per_image_and_reports_the_rest(tmp_path):
    from medproof.readers.export_seed import export_folder

    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for name, shade in (("a.png", 40), ("b.png", 90)):
        Image.new("L", (32, 32), shade).save(imgs / name)
    (imgs / "broken.png").write_bytes(b"not an image")
    out = tmp_path / "seed"
    written, problems = export_folder(reader(tmp_path / "c", Server()), imgs, "cxr", out)
    assert written == 2 and len(list(out.glob("*.json"))) == 2
    assert len(problems) == 1 and "broken.png" in problems[0]
