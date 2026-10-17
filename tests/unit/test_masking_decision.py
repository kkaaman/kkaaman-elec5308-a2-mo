"""BL-B-05: masking stays off unless detector-box evidence justifies its cost."""

from evaluation.masking import masking_decision


def pairs(n, plain, masked):
    return [dict(plain_error_m=plain, masked_error_m=masked) for _ in range(n)]


def test_clear_gain_within_budget_becomes_a_review_candidate():
    result = masking_decision(pairs(40, 0.1, 0.1), pairs(40, 0.9, 0.2), 0.012)
    assert result["decision"] == "CANDIDATE_ENABLE_REVIEW", result["reasons"]


def test_invalid_rate_reduction_also_counts_as_gain():
    detector = pairs(30, None, 0.3) + pairs(30, 0.3, 0.3)
    assert masking_decision([], detector, 0.01)["decision"] == "CANDIDATE_ENABLE_REVIEW"


def test_keep_off_when_gain_small_slow_or_unproven():
    assert masking_decision([], pairs(40, 0.3, 0.25), 0.01)["decision"] == "KEEP_OFF"
    assert masking_decision([], pairs(40, 0.9, 0.2), 0.08)["decision"] == "KEEP_OFF"
    assert masking_decision([], pairs(40, 0.9, 0.2), None)["decision"] == "KEEP_OFF"
    assert masking_decision([], pairs(5, 0.9, 0.2), 0.01)["decision"] == "KEEP_OFF"
    worse = pairs(40, 0.9, 0.1) + pairs(20, 0.3, None)
    assert masking_decision([], worse, 0.01)["decision"] == "KEEP_OFF"


def test_oracle_boxes_alone_never_enable_masking():
    result = masking_decision(pairs(100, 1.0, 0.1), [], 0.01)
    assert result["decision"] == "KEEP_OFF"
    assert result["oracle_boxes"]["median_abs_error_improvement_m"] > 0.8
