"""Raw CARLA BGRA camera decoding without importing the simulator."""

import numpy as np


def bgra_array(raw, width, height):
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Positive integer dimensions required")
    data = np.frombuffer(raw, dtype=np.uint8)
    if data.size != width * height * 4:
        raise ValueError("BGRA buffer size mismatch")
    return data.reshape(height, width, 4)


def decode_rgb(raw, width, height):
    return bgra_array(raw, width, height)[:, :, [2, 1, 0]].copy()


def decode_depth(raw, width, height):
    """Decoded CARLA values; axial versus radial convention is not inferred."""
    data = bgra_array(raw, width, height).astype(np.uint32)
    code = data[:, :, 2] + 256 * data[:, :, 1] + 65536 * data[:, :, 0]
    depth = (1000.0 * code / 16777215).astype(np.float32)
    depth[(code == 0) | (code == 16777215)] = np.nan
    return depth


def axial_depth(raw_depth, calibration, *, convention):
    """Caller must supply independently established installed-build convention."""
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
