"""Verify channels and invalid inputs independently of learned predictions."""
import numpy as np
import pytest
import torch

from autodrive.perception.lane_inference import LaneSegmenter as LegacySegmenter
from perception.lane_inference import LaneSegmenter


class ChannelProbe(torch.nn.Module):
    def forward(self, tensor):
        self.last_input = tensor.clone()
        # Label 1 iff red > blue, so a channel swap changes the result.
        difference = tensor[:, 0] - tensor[:, 2]
        return torch.stack((-difference, difference), dim=1)


def probe(cls=LaneSegmenter):
    model = cls.__new__(cls)
    model.image_size = (32, 64)
    model.device = torch.device("cpu")
    model.model = ChannelProbe().eval()
    return model


def test_rgb_is_not_swapped_and_normalization_is_correct():
    segmenter = probe()
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:] = [200, 80, 20]
    mask = segmenter.predict(rgb)
    assert mask.shape == (40, 80)
    assert mask.dtype == np.uint8
    assert np.all(mask == 1)
    actual = segmenter.model.last_input[0, :, 0, 0].numpy()
    np.testing.assert_allclose(actual, np.array([200, 80, 20]) / 255, atol=1e-7)


def test_legacy_bgr_and_canonical_rgb_receive_identical_tensors():
    rgb = np.zeros((40, 80, 3), dtype=np.uint8)
    rgb[:] = [200, 80, 20]
    canonical, legacy = probe(), probe(LegacySegmenter)
    np.testing.assert_array_equal(canonical.predict(rgb), legacy.predict(rgb[:, :, ::-1].copy()))
    torch.testing.assert_close(canonical.model.last_input, legacy.model.last_input)


@pytest.mark.parametrize("bad", [
    np.zeros((32, 32), dtype=np.uint8),
    np.zeros((32, 32, 4), dtype=np.uint8),
    np.zeros((32, 32, 3), dtype=np.float32),
    np.zeros((0, 32, 3), dtype=np.uint8),
])
def test_bad_images_rejected(bad):
    for cls in (LaneSegmenter, LegacySegmenter):
        with pytest.raises(ValueError):
            probe(cls).predict(bad)


def test_wrong_input_type_rejected():
    with pytest.raises(TypeError):
        probe().predict([1, 2, 3])


def test_constructor_uses_no_pretrained_download(monkeypatch):
    import perception.lane_inference as inference
    calls = []
    class FakeModel(ChannelProbe):
        def load_state_dict(self, state):
            calls.append(state)
    def build(**kwargs):
        assert kwargs == {"num_classes": 2, "pretrained": False}
        return FakeModel()
    def load(path, **kwargs):
        assert kwargs["weights_only"] is True
        return {"fixture": 1}
    monkeypatch.setattr(inference, "LaneUNet", build)
    monkeypatch.setattr(inference.torch, "load", load)
    segmenter = LaneSegmenter("fixture.pth", device="cpu")
    assert calls == [{"fixture": 1}]
    assert not segmenter.model.training
