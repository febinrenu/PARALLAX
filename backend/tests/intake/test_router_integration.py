"""How the router plugs into the intake stage. Uses a test double router; no model involved."""

from types import SimpleNamespace

import pytest

from medproof.intake.router import RouterOutput
from medproof.intake.run import run, run_bytes
from tests.conftest import make_phantom, png_bytes, to_u8


class TestDoubleRouter:
    __test__ = False

    def __init__(self, modality="cxr", confidence=0.95, error=None):
        self.modality, self.confidence, self.error = modality, confidence, error
        self.calls = 0

    def predict(self, img):
        self.calls += 1
        if self.error:
            raise self.error
        return RouterOutput(
            modality=self.modality, body_part=None, confidence=self.confidence,
            probs={self.modality: self.confidence}, method="probe",
        )


RAW = png_bytes(to_u8(make_phantom(256)))


def test_routed_modality_reaches_the_quality_gate_and_payload():
    res, _ = run_bytes(RAW, router=TestDoubleRouter("cxr"))
    assert res.ok
    assert res.payload["router"]["modality"] == "cxr" and res.payload["router"]["method"] == "probe"
    assert "tilt_deg" in res.payload["quality"]["metrics"]  # the CXR-only rotation check ran


def test_non_cxr_route_turns_the_rotation_check_off():
    res, _ = run_bytes(RAW, router=TestDoubleRouter("skin_dermoscopy"))
    assert res.payload["router"]["modality"] == "skin_dermoscopy"
    assert "tilt_deg" not in res.payload["quality"]["metrics"]


def test_explicit_hint_skips_routing():
    r = TestDoubleRouter("skin_dermoscopy")
    res, _ = run_bytes(RAW, modality_hint="cxr", router=r)
    assert r.calls == 0 and "router" not in res.payload
    assert "tilt_deg" in res.payload["quality"]["metrics"]


def test_router_failure_does_not_fail_intake():
    res, img = run_bytes(RAW, router=TestDoubleRouter(error=RuntimeError("model gone")))
    assert res.ok and img is not None
    assert any("router" in w.lower() for w in res.warnings)
    assert res.payload["router"] == {"error": "router_failed"}
    assert "tilt_deg" not in res.payload["quality"]["metrics"]


def test_low_confidence_route_is_not_trusted_for_cxr_only_checks():
    res, _ = run_bytes(RAW, router=TestDoubleRouter("cxr", confidence=0.30))
    assert "tilt_deg" not in res.payload["quality"]["metrics"]
    assert any("low router confidence" in w for w in res.warnings)


def test_no_router_means_unrouted_and_unchanged_behaviour():
    res, _ = run_bytes(RAW)
    assert res.ok and "router" not in res.payload


def test_run_uses_ctx_router_and_hint():
    ctx = SimpleNamespace(raw_bytes=RAW, router=TestDoubleRouter("cxr"))
    res = run(ctx)
    assert res.ok and res.payload["router"]["modality"] == "cxr"
    ctx2 = SimpleNamespace(raw_bytes=RAW, modality_hint="other", router=TestDoubleRouter("cxr"))
    assert "router" not in run(ctx2).payload


def test_bad_upload_still_fails_cleanly_with_a_router_present():
    r = TestDoubleRouter()
    res, img = run_bytes(b"garbage", router=r)
    assert res.ok is False and img is None and r.calls == 0
