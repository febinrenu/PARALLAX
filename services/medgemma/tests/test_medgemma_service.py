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


def test_truncated_reply_is_not_mistaken_for_an_inner_object():
    # Real failure: the reply was cut off mid-list and an inner {...} was returned as the result.
    cut = '```json\n{"labels": [{"name": "heart", "present": true}, {"name": "lung", "pres'
    with pytest.raises(ValueError):
        extract_json(cut)
    res = read_image(FakeBackend([cut, cut]), Image.new("L", (8, 8)), "cxr")
    assert not res.ok and "unparseable" in res.error


def test_object_without_expected_keys_is_unparseable():
    res = read_image(FakeBackend(['{"name": "heart"}', '{"foo": 1}']), Image.new("L", (8, 8)), "cxr")
    assert not res.ok


def test_prompt_asks_for_a_short_report_with_framing():
    be = FakeBackend([json.dumps(GOOD)])
    read_image(be, Image.new("L", (8, 8)), "cxr")
    p = be.prompts[0]
    assert "FINDINGS" in p and "IMPRESSION" in p and "research evaluation" in p.lower()


REPORT = (
    "FINDINGS: The lungs appear clear. No pleural effusion or pneumothorax is seen. "
    "The heart size is normal.\n\nIMPRESSION: No acute cardiopulmonary process identified."
)


def test_plain_report_is_accepted_without_json():
    res = read_image(FakeBackend([REPORT]), Image.new("L", (8, 8)), "cxr")
    assert res.ok and res.impression.startswith("No acute cardiopulmonary")
    assert "no pleural effusion" in res.findings_text.lower()
    assert res.labels == [] and res.boxes == []
    assert any("labels" in w for w in res.warnings)


def test_thinking_prefix_is_stripped():
    thought = "<unused94>thought\nThe user wants a report...<unused95>" + REPORT
    res = read_image(FakeBackend([thought]), Image.new("L", (8, 8)), "cxr")
    assert res.ok and "thought" not in res.findings_text and "user wants" not in res.findings_text


def test_unfinished_thinking_is_not_an_answer():
    unfinished = ["<unused94>thought\nThe user wants me to analyze", "<unused94>thought\nstill thinking"]
    res = read_image(FakeBackend(unfinished), Image.new("L", (8, 8)), "cxr")
    assert not res.ok


def test_refusal_is_retried_then_reported_not_ok():
    refusal = "I am unable to provide a medical diagnosis based on an image. I am an AI."
    be = FakeBackend([refusal, REPORT])
    res = read_image(be, Image.new("L", (8, 8)), "cxr")
    assert res.ok and len(be.prompts) == 2
    res2 = read_image(FakeBackend([refusal, refusal]), Image.new("L", (8, 8)), "cxr")
    assert not res2.ok and "refus" in res2.error


def test_compute_default_prefers_bf16_when_supported():
    from services.medgemma.loader import pick_compute

    assert pick_compute("auto", bf16_ok=True) == "bf16"
    assert pick_compute("auto", bf16_ok=False) == "fp32"
    assert pick_compute("fp16", bf16_ok=True) == "fp16"


def test_repeated_labels_and_boxes_are_collapsed_and_hedges_moved_out_of_names():
    # Real output: the same label five times, "possible" inside the name, near-identical boxes.
    rep = {
        "labels": [{"name": "possible pneumonia", "present": True, "confidence_text": "possible"}] * 5,
        "boxes": [
            {"label": "possible pneumonia", "box": [400, 300, 550, 700]},
            {"label": "possible pneumonia", "box": [450, 300, 550, 700]},
            {"label": "possible pneumonia", "box": [900, 10, 990, 80]},
        ],
        "impression": "Possible pneumonia.",
    }
    res = read_image(FakeBackend([json.dumps(rep)]), Image.new("L", (8, 8)), "cxr")
    assert [x.name for x in res.labels] == ["pneumonia"]
    assert res.labels[0].confidence_text == "possible"
    assert [b.label for b in res.boxes] == ["pneumonia", "pneumonia"]  # near-duplicate box merged
    assert res.boxes[0].xyxy_norm == (400, 300, 550, 700) and res.boxes[1].xyxy_norm == (900, 10, 990, 80)
    assert any("repeated" in w for w in res.warnings)


def test_label_with_likely_prefix_keeps_distinct_findings():
    rep = {"labels": [{"name": "likely nodule", "present": True}, {"name": "pleural effusion", "present": True}]}
    res = read_image(FakeBackend([json.dumps(rep)]), Image.new("L", (8, 8)), "cxr")
    assert [x.name for x in res.labels] == ["nodule", "pleural effusion"]
    assert res.labels[0].confidence_text == "likely"
