"""Raw CARLA BGRA camera decoding without importing the simulator."""

import numpy as np


def bgra_array(raw, width, height):
    """View a raw CARLA buffer as ``height x width x 4`` uint8 BGRA.

    ``width`` and ``height`` are positive integer pixels; the buffer must hold
    exactly ``width * height * 4`` bytes or ``ValueError`` is raised.
    """
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Positive integer dimensions required")
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size != width * height * 4:
        raise ValueError("BGRA buffer size mismatch")
    return data.reshape(height, width, 4)


def decode_rgb(raw, width, height):
    """Return a copied ``height x width x 3`` uint8 RGB image (alpha dropped)."""
    return bgra_array(raw, width, height)[:, :, [2, 1, 0]].copy()


def decode_depth(raw, width, height):
    """Decode CARLA 24-bit depth to float32 metres; convention is not inferred.

    ``depth = 1000 * (R + 256 G + 65536 B) / 16777215`` with channels promoted
    to uint32. Code 0 and the saturated 1000 m code are invalid and become NaN.
    Whether the result is axial optical depth or radial range must come from
    the installed-build calibration (see :func:`axial_depth`).
    """
    data = bgra_array(raw, width, height).astype(np.uint32)
    code = data[:, :, 2] + 256 * data[:, :, 1] + 65536 * data[:, :, 0]
    depth = (1000.0 * code / 16777215).astype(np.float32)
    depth[(code == 0) | (code == 16777215)] = np.nan
    return depth


def axial_depth(raw_depth, calibration, *, convention):
    """Return axial optical depth in metres from decoded CARLA depth.

    ``convention`` must be ``"axial"`` (values copied unchanged) or
    ``"radial"`` (each pixel divided by its normalised ray length using the
    calibration intrinsics in pixels). Any other value raises ``ValueError``;
    the convention must be established by ``scripts/calibrate_depth.py``, never
    assumed. NaN pixels stay NaN.
    """
    if convention == "axial":
        return raw_depth.copy()
    if convention != "radial":
        raise ValueError("Depth convention must be established as axial or radial")
    v, u = np.indices(raw_depth.shape)
    norm = np.sqrt(
        1
        + ((u - calibration.cx_px) / calibration.fx_px) ** 2
        + ((v - calibration.cy_px) / calibration.fy_px) ** 2
    )
    return (raw_depth / norm).astype(np.float32)
