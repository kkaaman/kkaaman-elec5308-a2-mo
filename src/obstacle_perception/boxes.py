"""RV-04 compatibility conversion; never invokes a detector."""

from obstacle_perception.types import PixelBox


def canonical_box(value):
    """Accept a legacy XYXY box or return an existing validated PixelBox."""
    if isinstance(value, PixelBox):
        return value
    try:
        return PixelBox(value.x_min, value.y_min, value.x_max, value.y_max)
    except AttributeError as error:
        raise TypeError("Expected an XYXY bounding box") from error
