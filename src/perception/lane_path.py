import cv2
import numpy as np


def _prepare_lane_mask(mask):
    """Clean a predicted lane mask before ego-lane extraction."""

    binary = (mask > 0).astype(np.uint8) * 255

    # Remove very small isolated detections.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= 20:
            cleaned[labels == label] = 255

    # Connect short vertical/diagonal gaps in broken lane markings.
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 15),
    )

    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        vertical_kernel,
    )

    # Suppress broad horizontal structures such as crosswalk stripes.
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (25, 3),
    )

    horizontal = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    cleaned = cv2.subtract(
        cleaned,
        horizontal,
    )

    return cleaned


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

    start_y = int(height * 0.90)
    end_y = int(height * 0.42)

    min_width = width * 0.05
    max_width = width * 0.75

    # Pass 1: collect every row's raw nearest-left/nearest-right pick
    # (as before), filtered only by width-plausibility -- this pass just
    # gathers evidence; outlier rejection needs a stable reference point,
    # which is what pass 2 establishes.
    raw_rows = []
    running_centre = width / 2
    for y in range(start_y, end_y, -4):
        clusters = _find_lane_clusters(mask, y)

        left_candidates = [x for x in clusters if x < running_centre]
        right_candidates = [x for x in clusters if x > running_centre]

        if not left_candidates or not right_candidates:
            continue

        left_x = max(left_candidates)
        right_x = min(right_candidates)
        lane_width = right_x - left_x

        if not (min_width <= lane_width <= max_width):
            continue

        raw_rows.append((y, left_x, right_x))
        running_centre = (
            0.7 * running_centre
            + 0.3 * ((left_x + right_x) / 2)
        )

    if len(raw_rows) < 6:
        return np.array([]), np.array([])

    # FIXED: reject outlier candidates against the GLOBAL MEDIAN left/right
    # position across the whole frame, not a temporally-sequential running
    # value. A sequential "reject jumps from the last accepted row"
    # approach is fragile to a leading or extended gap in the true
    # divider's paint (e.g. right over a crosswalk near the vehicle,
    # scanned first): once the very first row locks onto a farther, wrong
    # line (the retrained model -- see train_lane_segmentation.py -- now
    # reliably detects the OPPOSITE lane's double-yellow centreline, which
    # the old model rarely picked up), every subsequent row rejects the
    # TRUE divider as a "jump" relative to that wrong anchor, since the
    # wrong line keeps matching itself. The median is robust to this: as
    # long as the true divider dominates rows across the WHOLE frame
    # (a reasonable assumption even with an occasional gap), the median
    # reflects its real position regardless of which row happened to be
    # scanned first.
    median_left = float(np.median([row[1] for row in raw_rows]))
    median_right = float(np.median([row[2] for row in raw_rows]))
    max_jump_px = width * 0.15

    left_points = []
    right_points = []
    for y, left_x, right_x in raw_rows:
        if abs(left_x - median_left) > max_jump_px:
            continue
        if abs(right_x - median_right) > max_jump_px:
            continue
        left_points.append((left_x, y))
        right_points.append((right_x, y))

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
    """Extract ego-lane boundaries and an image-space centreline.

    ``mask`` is a finite 2-D numeric/boolean array where nonzero marks lane
    pixels. The result is a dict of ``left_boundary``, ``right_boundary`` and
    ``centerline`` arrays of ``(u, v)`` pixel coordinates (u right, v down),
    ``number_of_points`` rows ordered far-to-near. Returns ``None`` when a
    reliable lane pair cannot be found; thresholds are fixed pixel fractions,
    so accuracy on sharp curves, junctions or missing markings is not
    established. Pixels are converted to metres by the consumer, for example
    ``visualisation.lane_departure.lane_estimate_from_centerline``.
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

    if 0 in mask.shape:
        raise ValueError("Lane mask cannot be empty.")
    if not (np.issubdtype(mask.dtype, np.number) or mask.dtype == np.bool_):
        raise ValueError("Lane mask must be numeric or boolean.")
    if not np.isfinite(mask).all():
        raise ValueError("Lane mask must contain finite values.")
    if type(number_of_points) is not int or number_of_points < 6:
        raise ValueError("number_of_points must be an integer of at least 6.")

    height, width = mask.shape
    mask = _prepare_lane_mask(mask)

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
