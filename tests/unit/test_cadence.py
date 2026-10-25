"""TEST-17: rate-limited optional workloads never present a stale result as current."""

import pytest

from perception.cadence import CadenceResult, RateLimitedWorkload


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_runs_on_period_and_expires_between_computations():
    clock = FakeClock()
    workload = RateLimitedWorkload(3, clock=clock)
    calls = []

    def compute():
        calls.append(len(calls))
        clock.now += 0.02
        return f"mask-{len(calls)}"

    results = [workload.step(tick, compute) for tick in range(7)]
    assert [r.computed for r in results] == [True, False, False, True, False, False, True]
    assert [r.age_ticks for r in results] == [0, 1, 2, 0, 1, 2, 0]
    # Default max age is period - 1, so every returned value is valid and current.
    assert all(r.valid for r in results)
    assert [r.value for r in results] == ["mask-1"] * 3 + ["mask-2"] * 3 + ["mask-3"]
    assert results[0].cost_s == pytest.approx(0.02) and results[1].cost_s == 0.0
    assert workload.compute_count == 3 and workload.skip_count == 4
    assert workload.costs_s == pytest.approx([0.02, 0.02, 0.02])


def test_stale_result_is_invalid_and_not_returned_as_current():
    workload = RateLimitedWorkload(4, max_age_ticks=1, clock=FakeClock())
    first = workload.step(0, lambda: "mask-a")
    assert first == CadenceResult("mask-a", 0, True, True, None, 0.0)
    assert workload.step(1, lambda: "unused").valid
    stale = workload.step(2, lambda: "unused")
    assert not stale.valid
    assert stale.value is None
    assert stale.stale_value == "mask-a" and stale.age_ticks == 2
    # Non-contiguous ticks (dropped frames) age the result by elapsed ticks.
    later = workload.step(9, lambda: "mask-b")
    assert later.computed and later.valid and later.value == "mask-b" and later.age_ticks == 0


def test_before_first_computation_and_after_failure_nothing_is_current():
    workload = RateLimitedWorkload(2, clock=FakeClock())
    assert workload.due(0)
    workload.step(0, lambda: "ok")

    def broken():
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError):
        workload.step(2, broken)
    # The failed computation discarded the cache; the next tick recomputes.
    recovered = workload.step(3, lambda: "fresh")
    assert recovered.computed and recovered.value == "fresh"
    workload.reset()
    assert workload.compute_count == 0 and workload.due(0)


def test_validation():
    with pytest.raises(ValueError):
        RateLimitedWorkload(0)
    with pytest.raises(ValueError):
        RateLimitedWorkload(2, max_age_ticks=-1)
    with pytest.raises(ValueError):
        RateLimitedWorkload(True)
    workload = RateLimitedWorkload(2, clock=FakeClock())
    workload.step(5, lambda: 1)
    with pytest.raises(ValueError):
        workload.step(4, lambda: 1)
    with pytest.raises(ValueError):
        workload.step(-1, lambda: 1)
    with pytest.raises(ValueError):
        workload.step(1.5, lambda: 1)
