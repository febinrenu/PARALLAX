import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

from medproof.readers.cam import METHODS, CamExtractor, resize_cam  # noqa: E402


class Tiny(nn.Module):
    """features: ch0 = input pooled to 4x4, ch1 = constant. class0 reads ch0 only, class1 ch1 only."""

    def __init__(self):
        super().__init__()
        self.features = _Feat()
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def forward(self, x):
        f = F.relu(self.features(x))
        return self.classifier(F.adaptive_avg_pool2d(f, 1).flatten(1))


class _Feat(nn.Module):
    def forward(self, x):
        p = F.adaptive_avg_pool2d(x, 4)
        return torch.cat([p, torch.ones_like(p)], dim=1)


def _input(row, col):
    x = torch.zeros(1, 1, 64, 64)
    x[..., row * 16 : (row + 1) * 16, col * 16 : (col + 1) * 16] = 1.0
    return x


@pytest.mark.parametrize("method", METHODS)
def test_peak_is_where_the_class_evidence_is(method):
    model = Tiny().eval()
    with CamExtractor(model, model.features) as cx:
        cam = cx.maps(_input(1, 2), [0], method)[0]
    assert cam.shape == (4, 4) and cam.dtype == np.float32
    assert np.unravel_index(cam.argmax(), cam.shape) == (1, 2)
    assert cam.min() >= 0.0 and cam.max() == pytest.approx(1.0)
    assert (cam > 0).sum() == 1  # nothing else lights up


@pytest.mark.parametrize("method", METHODS)
def test_other_class_does_not_follow_the_input(method):
    model = Tiny().eval()
    with CamExtractor(model, model.features) as cx:
        a = cx.maps(_input(0, 0), [1], method)[0]
        b = cx.maps(_input(3, 3), [1], method)[0]
    np.testing.assert_allclose(a, b)  # class 1 only reads the constant channel


@pytest.mark.parametrize("method", METHODS)
def test_deterministic_and_batch_of_classes(method):
    model = Tiny().eval()
    x = _input(2, 1)
    with CamExtractor(model, model.features) as cx:
        one = cx.maps(x, [0], method)[0]
        both = cx.maps(x, [0, 1], method)
    assert len(both) == 2
    np.testing.assert_array_equal(one, both[0])


def test_hirescam_is_elementwise_gradient_times_activation():
    model = Tiny().eval()
    x = _input(1, 1)
    with CamExtractor(model, model.features) as cx:
        got = cx.maps(x, [0], "hirescam")[0]
    # d logit0 / d A0[h, w] = 1/16 everywhere, so the map is relu(A0 / 16), normalised
    a0 = F.adaptive_avg_pool2d(x, 4)[0, 0].numpy() / 16.0
    np.testing.assert_allclose(got, a0 / a0.max(), atol=1e-6)


def test_gradcam_uses_spatially_averaged_gradients():
    model = Tiny().eval()
    x = _input(0, 3) * 0.5 + _input(3, 0)
    with CamExtractor(model, model.features) as cx:
        got = cx.maps(x, [0], "gradcam")[0]
    a0 = F.adaptive_avg_pool2d(x, 4)[0, 0].numpy()
    np.testing.assert_allclose(got, a0 / a0.max(), atol=1e-6)  # alpha0 is a positive constant


def test_constant_signal_gives_all_zero_map_not_nan():
    model = Tiny().eval()
    with CamExtractor(model, model.features) as cx:
        cam = cx.maps(torch.zeros(1, 1, 64, 64), [0], "gradcam++")[0]
    assert np.isfinite(cam).all() and cam.max() == 0.0


def test_unknown_method_and_hooks_are_cleaned_up():
    model = Tiny().eval()
    with CamExtractor(model, model.features) as cx:
        with pytest.raises(ValueError):
            cx.maps(_input(0, 0), [0], "nope")
    assert len(model.features._forward_hooks) == 0


def test_resize_cam_maps_to_original_grid_and_keeps_range():
    cam = np.zeros((4, 4), np.float32)
    cam[1, 2] = 1.0
    out = resize_cam(cam, (300, 500))  # height, width of the original image
    assert out.shape == (300, 500) and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-6
    y, x = np.unravel_index(out.argmax(), out.shape)
    assert 300 * 1 / 4 <= y <= 300 * 2 / 4 and 500 * 2 / 4 <= x <= 500 * 3 / 4


@pytest.mark.parametrize("method", METHODS)
def test_negative_pre_relu_activations_do_not_invert_the_map(method):
    """A backbone hooked before its final ReLU can have large negative background values."""
    model = Tiny().eval()
    x = _input(1, 2) * 2.0 - 1.0  # background -1, blob +1
    with CamExtractor(model, model.features) as cx:
        cam = cx.maps(x, [0], method)[0]
    assert np.unravel_index(cam.argmax(), cam.shape) == (1, 2)
    assert (cam > 0).sum() == 1
