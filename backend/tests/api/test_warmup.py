"""Startup warm-up: load the heavy models in the background so the first study is not the slow one."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from medproof.api import warmup
from medproof.api.app import create_app
from medproof.core.config import PipelineConfig


def test_parse_accepts_on_off_and_a_list():
    assert warmup.parse_targets("1") == warmup.ALL_TARGETS
    assert warmup.parse_targets("all") == warmup.ALL_TARGETS
    assert warmup.parse_targets("0") == ()
    assert warmup.parse_targets("") == ()
    assert warmup.parse_targets(None) == warmup.ALL_TARGETS  # on unless switched off
    assert warmup.parse_targets("cxr, precedents") == ("cxr", "precedents")
    assert warmup.parse_targets("CXR;nonsense") == ("cxr",)  # unknown names are ignored, not fatal


def test_every_target_loads_through_the_loader_and_its_time_is_recorded():
    loaded = []
    loaders = {t: (lambda t=t: loaded.append(t)) for t in warmup.ALL_TARGETS}
    out = warmup.warm_up(warmup.ALL_TARGETS, loaders=loaders)
    assert loaded == list(warmup.ALL_TARGETS)
    assert set(out) == set(warmup.ALL_TARGETS) and all(v["ok"] and v["seconds"] >= 0 for v in out.values())


def test_a_loader_that_fails_is_recorded_and_the_rest_still_load():
    def boom():
        raise RuntimeError("weights missing")

    loaded = []
    loaders = {"cxr": boom, "bone_xray": lambda: loaded.append("bone_xray"), "precedents": lambda: loaded.append("precedents")}
    out = warmup.warm_up(("cxr", "bone_xray", "precedents"), loaders=loaders)
    assert loaded == ["bone_xray", "precedents"]
    assert out["cxr"]["ok"] is False and "RuntimeError" in out["cxr"]["error"] and "weights missing" in out["cxr"]["error"]
    assert out["bone_xray"]["ok"] and out["precedents"]["ok"]


def test_a_target_with_no_loader_is_reported_not_raised():
    out = warmup.warm_up(("cxr",), loaders={})
    assert out["cxr"]["ok"] is False


def test_the_app_does_not_warm_up_unless_asked(monkeypatch):
    calls = []
    monkeypatch.setattr(warmup, "warm_up", lambda *a, **k: calls.append(1) or {})
    app = create_app(PipelineConfig())
    with TestClient(app):
        pass
    assert calls == []


def test_the_app_warms_up_in_the_background_when_asked(monkeypatch):
    started, release = threading.Event(), threading.Event()

    def slow_warm_up(targets, **kwargs):
        started.set()
        release.wait(5)
        return {t: {"ok": True, "seconds": 0.0} for t in targets}

    monkeypatch.setattr(warmup, "warm_up", slow_warm_up)
    app = create_app(PipelineConfig(), warmup_targets=("cxr",))
    with TestClient(app) as client:
        assert started.wait(5)  # the warm-up is running ...
        assert client.get("/metrics").status_code == 200  # ... and the server already answers
        release.set()
    # the result is kept for anyone who wants to look
    assert app.state.warmup_thread is not None


def test_a_crashing_warm_up_never_takes_the_server_down(monkeypatch):
    def crash(*a, **k):
        raise MemoryError("out of memory")

    monkeypatch.setattr(warmup, "warm_up", crash)
    app = create_app(PipelineConfig(), warmup_targets=("cxr",))
    with TestClient(app) as client:
        app.state.warmup_thread.join(5)
        assert client.get("/metrics").status_code == 200


@pytest.mark.parametrize("env,expected", [("0", ()), ("1", warmup.ALL_TARGETS), ("cxr", ("cxr",))])
def test_the_environment_variable_controls_the_default_for_the_real_app(monkeypatch, env, expected):
    monkeypatch.setenv("MEDPROOF_WARMUP", env)
    assert warmup.targets_from_env() == expected


def test_default_loaders_cover_every_target():
    loaders = warmup.default_loaders(SimpleNamespace(config=PipelineConfig()))
    assert set(loaders) == set(warmup.ALL_TARGETS)


# ---- the second reader runs in another process that loads its model on the first request, so warming means making one tiny request

def test_the_second_reader_is_a_warm_up_target_and_loads_last():
    assert "second_read" in warmup.ALL_TARGETS and warmup.ALL_TARGETS[-1] == "second_read"
    assert warmup.parse_targets("cxr,second_read") == ("cxr", "second_read")


def test_the_second_reader_warm_up_makes_one_small_request_to_the_service(monkeypatch):
    import httpx

    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        return httpx.Response(200, json={"ok": True, "model": "m", "impression": "x", "findings_text": "x", "labels": [], "boxes": []})

    monkeypatch.setenv("MEDGEMMA_URL", "http://service.invalid")
    monkeypatch.setattr(warmup, "_service_transport", lambda: httpx.MockTransport(handler))
    warmup.default_loaders(SimpleNamespace(config=PipelineConfig()))["second_read"]()
    assert seen == [("POST", "/read")]


def test_without_a_service_address_the_second_reader_warm_up_is_reported_not_raised(monkeypatch):
    monkeypatch.delenv("MEDGEMMA_URL", raising=False)
    out = warmup.warm_up(("second_read",), loaders=warmup.default_loaders(SimpleNamespace(config=PipelineConfig())))
    assert out["second_read"]["ok"] is False and "MEDGEMMA_URL" in out["second_read"]["error"]


def test_an_unavailable_service_is_reported_as_a_failed_warm_up(monkeypatch):
    import httpx

    monkeypatch.setenv("MEDGEMMA_URL", "http://service.invalid")
    monkeypatch.setattr(warmup, "_service_transport", lambda: httpx.MockTransport(lambda r: httpx.Response(503, text="down")))
    out = warmup.warm_up(("second_read",), loaders=warmup.default_loaders(SimpleNamespace(config=PipelineConfig())))
    assert out["second_read"]["ok"] is False
