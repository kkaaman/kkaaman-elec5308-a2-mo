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


def estimate_detection_depth(depth_m, bounding_box, calibration, policy=None):
    """Aligned axial-depth surface estimate with explicit validity diagnostics.

    Unlike the legacy scalar pinhole helper, this does not assume object size.
    An invalid result represents unknown depth, never road clearance.
    """
    from obstacle_perception.box_depth import estimate_box_depth

    return estimate_box_depth(depth_m, bounding_box, calibration, policy)
