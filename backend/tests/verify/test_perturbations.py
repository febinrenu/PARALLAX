import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from medproof.verify import perturbations as P
from tests.conftest import make_phantom


@pytest.fixture(scope="module")
def img():
    return make_phantom(256)


def test_eight_named_perturbations():
    assert len(P.NAMES) == 8
    assert set(P.NAMES) == {"noise", "contrast", "gamma", "jpeg", "rotate", "downsample", "blur", "crop"}


@pytest.mark.parametrize("name", P.NAMES)
@pytest.mark.parametrize("sev", P.SEVERITIES)
def test_shape_dtype_range_preserved(img, name, sev):
    out = P.apply(name, img, sev, seed=1)
    assert out.shape == img.shape and out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


@pytest.mark.parametrize("name", P.NAMES)
def test_deterministic_for_same_seed(img, name):
    np.testing.assert_array_equal(P.apply(name, img, 3, seed=5), P.apply(name, img, 3, seed=5))


def test_seed_changes_stochastic_ones(img):
    assert not np.array_equal(P.apply("noise", img, 3, seed=1), P.apply("noise", img, 3, seed=2))


@pytest.mark.parametrize("name", P.NAMES)
def test_damage_grows_with_severity(img, name):
    errs = [float(np.mean((P.apply(name, img, s, seed=0) - img) ** 2)) for s in P.SEVERITIES]
    assert errs[-1] > errs[0], (name, errs)
    assert all(b >= a - 1e-6 for a, b in zip(errs, errs[1:])), (name, errs)


def test_colour_images_supported(img):
    rgb = np.stack([img, img * 0.8, img * 0.6], axis=2).astype(np.float32)
    for name in P.NAMES:
        assert P.apply(name, rgb, 2).shape == rgb.shape


def test_input_is_not_modified(img):
    before = img.copy()
    for name in P.NAMES:
        P.apply(name, img, 4)
    np.testing.assert_array_equal(img, before)


def test_bad_arguments():
    with pytest.raises(KeyError):
        P.apply("nope", np.zeros((8, 8), np.float32))
    with pytest.raises(ValueError):
        P.apply("noise", np.zeros((8, 8), np.float32), severity=9)


@settings(max_examples=20, deadline=None)
@given(st.sampled_from(P.NAMES), st.sampled_from(P.SEVERITIES), st.integers(0, 10_000))
def test_any_combination_is_valid_property(name, sev, seed):
    base = np.random.default_rng(seed).random((40, 56)).astype(np.float32)
    out = P.apply(name, base, sev, seed)
    assert out.shape == base.shape and np.isfinite(out).all()
