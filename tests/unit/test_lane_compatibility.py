import numpy as np
import pytest

from autodrive.perception.lane_path import extract_ego_path as legacy_extract
from perception.lane_path import extract_ego_path


def make_mask(curved=False, broken=False):
    mask = np.zeros((540, 960), dtype=np.uint8)
    for y in range(230, 490):
        if broken and (y // 25) % 2:
            continue
        shift = int(0.0007 * (y - 480) ** 2) if curved else 0
        mask[y, 300 + shift:306 + shift] = 1
        mask[y, 654 + shift:660 + shift] = 1
    return mask


def test_legacy_import_uses_canonical_function():
    assert legacy_extract is extract_ego_path


@pytest.mark.parametrize("curved,broken", [
    (False, False), (True, False), (False, True),
])
def test_representative_lane_centres(curved, broken):
    result = extract_ego_path(make_mask(curved, broken))
    assert result is not None
    center = result["centerline"]
    expected = 479.5 + (
        0.0007 * (center[:, 1] - 480) ** 2 if curved else 0
    )
    assert np.allclose(center[:, 0], expected, atol=5)


def test_single_boundary_is_not_a_lane_pair():
    mask = make_mask()
    mask[:, 600:] = 0
    assert extract_ego_path(mask) is None
