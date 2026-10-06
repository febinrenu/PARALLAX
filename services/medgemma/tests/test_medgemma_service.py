import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from services.medgemma.batch import run_batch
from services.medgemma.reader import read_image
from services.medgemma.schema import extract_json
from services.medgemma.server import create_app

GOOD = {
    "labels": [{"name": "consolidation", "present": True, "confidence_text": "likely"}],
    "boxes": [{"label": "consolidation", "box": [600, 500, 900, 800]}],
    "impression": "Right lower zone opacity.",
}


class FakeBackend:
    name = "fake"
    quant = "none"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, image, prompt, max_new_tokens=700):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def png_bytes(size=(64, 48)):
    buf = io.BytesIO()
    Image.new("L", size, 128).save(buf, "PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here you go: " + json.dumps(GOOD) + " hope this helps",
    ],
)
def test_extract_json_tolerant(text):
    assert extract_json(text)["impression"] == GOOD["impression"]


def test_extract_json_ignores_braces_inside_strings():
    text = '{"impression": "a } b", "labels": [], "boxes": []}'
    assert extract_json(text)["impression"] == "a } b"


def test_extract_json_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_read_happy_path_normalizes_boxes():
    be = FakeBackend([json.dumps(GOOD)])
    res = read_image(be, Image.new("L", (64, 48)), "cxr")
    assert res.ok and res.labels[0].name == "consolidation"
    assert res.boxes[0].xyxy_norm == (600, 500, 900, 800)
    assert res.impression


def test_invalid_boxes_are_dropped_with_warning():
    bad = {**GOOD, "boxes": [{"label": "x", "box": [900, 10, 100, 20]}, {"label": "y", "box": [0, 0, 5000, 10]}]}
    res = read_image(FakeBackend([json.dumps(bad)]), Image.new("L", (8, 8)), "cxr")
    assert res.ok and res.boxes == [] and len(res.warnings) == 2


def test_repair_retry_then_success():
    be = FakeBackend(["sorry, I cannot", json.dumps(GOOD)])
    res = read_image(be, Image.new("L", (8, 8)), "cxr")
    assert res.ok and len(be.prompts) == 2 and "only" in be.prompts[1].lower()


def test_failure_returns_ok_false_with_raw():
    be = FakeBackend(["nope", "still nope"])
    res = read_image(be, Image.new("L", (8, 8)), "cxr")
    assert not res.ok and res.raw == "still nope" and res.error


def test_backend_exception_does_not_crash():
    class Boom(FakeBackend):
        def generate(self, *a, **k):
            raise RuntimeError("cuda out of memory")

    res = read_image(Boom([]), Image.new("L", (8, 8)), "cxr")
    assert not res.ok and "out of memory" in res.error


def test_non_cxr_prompt_does_not_ask_for_boxes():
    be = FakeBackend([json.dumps({**GOOD, "boxes": []})])
    read_image(be, Image.new("L", (8, 8)), "skin_dermoscopy")
    assert "box" not in be.prompts[0].lower()


def test_api_health_and_read():
    client = TestClient(create_app(FakeBackend([json.dumps(GOOD)])))
    assert client.get("/health").json()["loaded"] is True
    r = client.post(
        "/read",
        files={"image": ("a.png", png_bytes(), "image/png")},
        data={"modality": "cxr"},
    )
    body = r.json()
    assert r.status_code == 200 and body["ok"] and body["boxes"][0]["label"] == "consolidation"


def test_api_rejects_non_image():
    client = TestClient(create_app(FakeBackend([])))
    r = client.post("/read", files={"image": ("a.txt", b"hello", "text/plain")}, data={"modality": "cxr"})
    assert r.status_code == 400


def test_api_rejects_unknown_modality():
    client = TestClient(create_app(FakeBackend([])))
    r = client.post("/read", files={"image": ("a.png", png_bytes(), "image/png")}, data={"modality": "ct"})
    assert r.status_code == 422


def test_batch_is_resumable(tmp_path):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for n in ("a", "b", "c"):
        (imgs / f"{n}.png").write_bytes(png_bytes())
    out = tmp_path / "reads.jsonl"
    be = FakeBackend([json.dumps(GOOD)] * 3)
    run_batch(be, imgs, out, modality="cxr", limit=2)
    assert len(out.read_text().splitlines()) == 2
    run_batch(be, imgs, out, modality="cxr")
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 3 and len({row["sha256"] for row in lines}) >= 1
    assert len({row["image"] for row in lines}) == 3
    assert len(be.prompts) == 3  # nothing was re-read


@pytest.mark.gpu
def test_real_model_smoke():
    pytest.skip("run `python -m services.medgemma.smoke --image <cxr>` on a GPU machine")
