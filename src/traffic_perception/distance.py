from ._validation import positive


def estimate_distance_m(
    observed_width_px: float,
    real_width_m: float,
    focal_length_px: float,
) -> float:
    """Estimate object distance using the pinhole-camera model."""
    positive("observed_width_px", observed_width_px)
    positive("real_width_m", real_width_m)
    positive("focal_length_px", focal_length_px)

    return focal_length_px * real_width_m / observed_width_px
