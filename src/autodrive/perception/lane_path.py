import cv2
import numpy as np


def _find_lane_clusters(mask, y, band_height=3):
    """Find x-coordinate centres of lane-pixel groups near row y."""

    height, _ = mask.shape

    y_start = max(0, y - band_height)
    y_end = min(height, y + band_height + 1)

    band = mask[y_start:y_end]

    active_x = np.where(
        np.any(band > 0, axis=0)
    )[0]

    if len(active_x) == 0:
        return []

    groups = []
    current_group = [active_x[0]]

    for x in active_x[1:]:
        if x - current_group[-1] <= 12:
            current_group.append(x)
        else:
            groups.append(current_group)
            current_group = [x]

    groups.append(current_group)

    return [
        int(np.mean(group))
        for group in groups
    ]


def _collect_ego_lane_points(mask):
    """Track the nearest left and right lane boundaries."""

    height, width = mask.shape

    current_centre = width / 2

    left_points = []
    right_points = []

    start_y = int(height * 0.90)
    end_y = int(height * 0.42)

    for y in range(start_y, end_y, -4):
        clusters = _find_lane_clusters(
            mask,
            y,
        )

        left_candidates = [
            x for x in clusters
            if x < current_centre
        ]

        right_candidates = [
            x for x in clusters
            if x > current_centre
        ]

        if (
            not left_candidates
            or not right_candidates
        ):
            continue

        left_x = max(left_candidates)
        right_x = min(right_candidates)

        lane_width = right_x - left_x

        min_width = width * 0.05
        max_width = width * 0.75

        if not (
            min_width <= lane_width <= max_width
        ):
            continue

        left_points.append(
            (left_x, y)
        )

        right_points.append(
            (right_x, y)
        )

        detected_centre = (
            left_x + right_x
        ) / 2

        current_centre = (
            0.7 * current_centre
            + 0.3 * detected_centre
        )

    return (
        np.array(left_points),
        np.array(right_points),
    )


def _fit_boundary(points):
    """Fit a robust quadratic x=f(y) lane boundary."""

    if len(points) < 6:
        return None

    x = points[:, 0]
    y = points[:, 1]

    for _ in range(3):
        model = np.polyfit(
            y,
            x,
            2,
        )

        predicted_x = np.polyval(
            model,
            y,
        )

        errors = np.abs(
            x - predicted_x
        )

        median_error = np.median(
            errors
        )

        threshold = max(
            8,
            median_error * 2.5,
        )

        keep = errors <= threshold

        x = x[keep]
        y = y[keep]

        if len(x) < 6:
            return None

    return np.polyfit(
        y,
        x,
        2,
    )


def extract_ego_path(
    mask,
    number_of_points=100,
):
    """
    Extract ego-lane boundaries and an image-space centreline.

    Returns None if a reliable lane pair cannot be found.
    """

    if mask is None:
        raise ValueError(
            "Lane mask cannot be None."
        )

    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
    ):
        raise ValueError(
            "Lane mask must be a 2D NumPy array."
        )

    height, width = mask.shape

    left_points, right_points = (
        _collect_ego_lane_points(mask)
    )

    if (
        len(left_points) < 6
        or len(right_points) < 6
    ):
        return None

    left_model = _fit_boundary(
        left_points
    )

    right_model = _fit_boundary(
        right_points
    )

    if (
        left_model is None
        or right_model is None
    ):
        return None

    highest_supported_y = max(
        left_points[:, 1].min(),
        right_points[:, 1].min(),
    )

    lowest_supported_y = min(
        left_points[:, 1].max(),
        right_points[:, 1].max(),
    )

    if highest_supported_y >= lowest_supported_y:
        return None

    y_values = np.linspace(
        highest_supported_y,
        lowest_supported_y,
        number_of_points,
    )

    left_x = np.polyval(
        left_model,
        y_values,
    )

    right_x = np.polyval(
        right_model,
        y_values,
    )

    valid = (
        (left_x >= 0)
        & (left_x < width)
        & (right_x >= 0)
        & (right_x < width)
        & (left_x < right_x)
    )

    left_x = left_x[valid]
    right_x = right_x[valid]
    y_values = y_values[valid]

    if len(y_values) < 6:
        return None

    centre_x = (
        left_x + right_x
    ) / 2

    left_boundary = np.column_stack(
        (left_x, y_values)
    )

    right_boundary = np.column_stack(
        (right_x, y_values)
    )

    centerline = np.column_stack(
        (centre_x, y_values)
    )

    return {
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "centerline": centerline,
    }


def draw_ego_path(
    image,
    path,
    alpha=0.35,
):
    """Draw ego-lane boundaries, corridor, and centreline."""

    if path is None:
        return image.copy()

    output = image.copy()
    overlay = image.copy()

    left = path["left_boundary"]
    right = path["right_boundary"]
    center = path["centerline"]

    polygon = np.vstack(
        (
            left,
            right[::-1],
        )
    ).astype(np.int32)

    cv2.fillPoly(
        overlay,
        [polygon],
        (255, 100, 0),
    )

    output = cv2.addWeighted(
        overlay,
        alpha,
        output,
        1 - alpha,
        0,
    )

    left_int = left.astype(np.int32)
    right_int = right.astype(np.int32)
    center_int = center.astype(np.int32)

    cv2.polylines(
        output,
        [left_int],
        False,
        (0, 255, 0),
        3,
    )

    cv2.polylines(
        output,
        [right_int],
        False,
        (0, 255, 0),
        3,
    )

    cv2.polylines(
        output,
        [center_int],
        False,
        (0, 0, 255),
        3,
    )

    return output