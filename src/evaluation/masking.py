"""BL-B-05 decision rule for optional semantic masking of box depth.

Masking stays OFF unless measured evidence justifies its cost. The decision
uses detector boxes (what the runtime actually sees); oracle visible boxes are
reported for context only, because tight ground-truth boxes contain little
background and understate what masking could remove.
"""

import math

import numpy as np


def _summary(pairs):
    rows = list(pairs)
    paired = [r for r in rows if r["plain_error_m"] is not None and r["masked_error_m"] is not None]
    improvement = (
        float(np.median([abs(r["plain_error_m"]) - abs(r["masked_error_m"]) for r in paired]))
        if paired
        else None
    )
    return dict(
        boxes=len(rows),
        paired_valid=len(paired),
        median_abs_error_improvement_m=improvement,
        plain_invalid_rate=(sum(r["plain_error_m"] is None for r in rows) / len(rows))
        if rows
        else None,
        masked_invalid_rate=(sum(r["masked_error_m"] is None for r in rows) / len(rows))
        if rows
        else None,
    )


def masking_decision(
    oracle_pairs,
    detector_pairs,
    mask_warm_p95_s,
    *,
    latency_budget_s=0.03,
    minimum_pairs=30,
    minimum_improvement_m=0.2,
    minimum_invalid_reduction=0.10,
):
    """Return per-box-source summaries and KEEP_OFF / CANDIDATE_ENABLE_REVIEW."""
    oracle, detector = _summary(oracle_pairs), _summary(detector_pairs)
    reasons = []
    if detector["boxes"] < minimum_pairs:
        reasons.append(f"fewer than {minimum_pairs} detector boxes matched to truth")
    if mask_warm_p95_s is None or not math.isfinite(mask_warm_p95_s):
        reasons.append("segmentation runtime not measured")
    elif mask_warm_p95_s > latency_budget_s:
        reasons.append(
            f"mask warm p95 {mask_warm_p95_s * 1000:.0f} ms exceeds {latency_budget_s * 1000:.0f} ms"
        )
    gain = False
    if detector["boxes"]:
        improvement = detector["median_abs_error_improvement_m"]
        reduction = detector["plain_invalid_rate"] - detector["masked_invalid_rate"]
        if detector["masked_invalid_rate"] > detector["plain_invalid_rate"]:
            reasons.append("masking increases the invalid-depth rate")
        gain = (improvement is not None and improvement >= minimum_improvement_m) or (
            reduction >= minimum_invalid_reduction
        )
        if not gain:
            reasons.append("measured depth/invalid-rate gain below the threshold")
    decision = "CANDIDATE_ENABLE_REVIEW" if gain and not reasons else "KEEP_OFF"
    return dict(
        decision=decision,
        reasons=reasons,
        detector_boxes=detector,
        oracle_boxes=oracle,
        thresholds=dict(
            latency_budget_s=latency_budget_s,
            minimum_pairs=minimum_pairs,
            minimum_improvement_m=minimum_improvement_m,
            minimum_invalid_reduction=minimum_invalid_reduction,
        ),
        stale_mask_policy=(
            "A mask is only applied to depth from the same frame. Runtime masking"
            " stays disabled unless a reviewed decision enables it."
        ),
    )
