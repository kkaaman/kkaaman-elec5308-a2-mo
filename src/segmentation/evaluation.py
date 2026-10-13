"""Metrics for comparing segmentation backends against ground-truth trainId masks.

Predictions of 255 (unlabelled) are counted as misses, not ignored, so an
objects-only model such as YOLO-seg is penalised for every object pixel it fails
to label. Ground-truth pixels of 255 are excluded, following Cityscapes practice.
"""

from __future__ import annotations

import numpy as np

from .labels import CITYSCAPES_CLASSES, IGNORE_ID, OBSTACLE_TRAIN_IDS

NUM_CLASSES = len(CITYSCAPES_CLASSES)


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return a ``num_classes x (num_classes + 1)`` confusion matrix.

    Rows are ground-truth classes. Columns are predicted classes, with the last
    column counting pixels the model left unlabelled (255 or any id outside the
    class range).
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.shape != pred.shape:
        raise ValueError(f"gt shape {gt.shape} != prediction shape {pred.shape}")
    valid = (gt != IGNORE_ID) & (gt < num_classes)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    p[(p < 0) | (p >= num_classes)] = num_classes
    counts = np.bincount(g * (num_classes + 1) + p, minlength=num_classes * (num_classes + 1))
    return counts.reshape(num_classes, num_classes + 1)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """IoU for each class; NaN for classes absent from the ground truth."""
    n = cm.shape[0]
    tp = np.diag(cm[:, :n]).astype(np.float64)
    gt_total = cm.sum(axis=1).astype(np.float64)
    pred_total = cm[:, :n].sum(axis=0).astype(np.float64)
    union = gt_total + pred_total - tp
    iou = np.full(n, np.nan)
    present = gt_total > 0
    iou[present] = tp[present] / union[present]
    return iou


def summarize(cm: np.ndarray, labels_all_classes: bool = True) -> dict:
    """Headline metrics from an accumulated confusion matrix.

    Args:
        cm: Matrix from :func:`confusion_matrix`, summed over frames.
        labels_all_classes: False for objects-only models; their 19-class mIoU
            and pixel accuracy are then reported as None because they would
            measure classes the model never predicts.
    """
    iou = per_class_iou(cm)
    obstacle = iou[list(OBSTACLE_TRAIN_IDS)]
    total = cm.sum()
    result = {
        "obstacle_miou": _nanmean(obstacle),
        "miou_19": _nanmean(iou) if labels_all_classes else None,
        "pixel_accuracy": (float(np.trace(cm[:, : cm.shape[0]]) / total) if total and labels_all_classes else None),
        "per_class_iou": {
            CITYSCAPES_CLASSES[i]: (None if np.isnan(v) else round(float(v), 4)) for i, v in enumerate(iou)
        },
        "evaluated_pixels": int(total),
    }
    for key in ("obstacle_miou", "miou_19", "pixel_accuracy"):
        if result[key] is not None:
            result[key] = round(result[key], 4)
    return result


def _nanmean(values: np.ndarray) -> float | None:
    values = values[~np.isnan(values)]
    return float(values.mean()) if values.size else None
