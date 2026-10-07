"""The reader stage leaves what faithfulness and stability need on the shared study context."""

import pytest

torch = pytest.importorskip("torch")

from medproof.core.context import StudyContext  # noqa: E402
from medproof.intake.decode import load_image  # noqa: E402
from medproof.readers import cxr  # noqa: E402
from medproof.verify import faithfulness as FA  # noqa: E402
from medproof.verify import stability as ST  # noqa: E402
from tests.conftest import png_bytes, to_u8  # noqa: E402
from tests.readers.test_cxr import _image, make_reader  # noqa: E402


def _ctx(tmp_path):
    img = load_image(png_bytes(to_u8(_image())))
    return StudyContext(study_id="s1", decoded=img, artifact_dir=tmp_path)


def test_reader_stage_shares_reader_output_and_findings_with_later_stages(tmp_path):
    ctx = _ctx(tmp_path)
    res = cxr.run(ctx, reader=make_reader(sharpness=6.0))
    assert res.ok
    assert ctx.reader_output is not None and ctx.reader is not None
    assert [f.label for f in ctx.findings] == [f["label"] for f in res.payload["findings"]]


def test_faithfulness_and_stability_run_on_the_shared_context(tmp_path):
    ctx = _ctx(tmp_path)
    reader = make_reader(sharpness=6.0)
    cxr.run(ctx, reader=reader)
    fa = FA.run(ctx, reader=ctx.reader, cfg=FA.FaithfulnessConfig(n_random=12))
    st = ST.run(ctx, reader=ctx.reader)
    assert fa.ok and st.ok
    assert {f["label"] for f in fa.payload["findings"]} == {f.label for f in ctx.findings}
    assert all(f["stability"]["tests"] == 8 for f in st.payload["findings"])


def test_context_that_cannot_take_attributes_does_not_break_the_stage():
    class Frozen:
        __slots__ = ("decoded", "artifact_dir")

        def __init__(self, decoded):
            self.decoded, self.artifact_dir = decoded, None

    res = cxr.run(Frozen(load_image(png_bytes(to_u8(_image())))), reader=make_reader())
    assert res.ok
