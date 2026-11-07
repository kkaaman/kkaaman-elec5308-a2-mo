"""Extend TEST-18 with production bounded workers and hybrid stack integration.

Synthetic sensor/backend inputs; worker completion is awaited by the TEST
orchestrator before the real nonblocking control step, never inside that step.
No HTTP, real-model latency, physical or extension admission claim.
"""

import argparse
import itertools
import json
from types import SimpleNamespace

from autonomy.hybrid_detector import HybridFrameDetector
from autonomy.interfaces import LaneGraph
from autonomy.stack import SkeletonStack
from perception.async_detector import AsyncDetector
from perception.hybrid_detector import HybridDetector
from scripts.offline_fault_suite import (
    BEHAVIOURS,
    FAULTS,
    FaultFrameDetector,
    FixtureBackend,
    Rig,
    run_case,
    write_report,
)


class HybridRig(Rig):
    def __init__(self, profile, behaviour, tracking):
        super().__init__("local_fixture", behaviour, tracking)
        self.stack.close()
        self.backend = SimpleNamespace(fault=None, closed=False, reset=lambda **kw: None)
        self.locals = FixtureBackend(self.clock, "local_fixture")
        self.remotes = FixtureBackend(self.clock, "local_fixture")
        local = FaultFrameDetector(
            self.locals, source_id="fixture-front", model_version="fixture-v1"
        )
        remote = FaultFrameDetector(
            self.remotes, source_id="fixture-front", model_version="fixture-v1"
        )
        self.hybrid = HybridDetector(
            local, AsyncDetector(remote, clock=self.clock), clock=self.clock
        )
        self.detector = HybridFrameDetector(self.hybrid, self.policy, model_version="fixture-v1")
        self.detector.last_result = None
        self.stack = SkeletonStack(
            self.detector, self.config, self.policy, clock=self.clock,
            freshness_gate=self.gate, simulation_clock=lambda: self.clock.sim,
            use_tracking=tracking)
        self.profile = profile
        self.stack.reset(episode_id="test18-session", graph=LaneGraph((), ()))

    def prepare(self, frame):
        self.locals.fault = self.remotes.fault = self.backend.fault
        if self.profile == "remote":
            self.locals.fault = "detector_exception"
        if self.backend.fault == "generation_restart":
            self.detector.generation += 1
            self.hybrid.reset(episode_id="test18-session")
        # Invalid identity/state is rejected by the real stack before inference.
        if frame.episode_id != "test18-session":
            return
        self.detector.submit(frame)
        for worker in (self.hybrid.local, self.hybrid.remote):
            if worker.work is not None and not worker.work.done.wait(1):
                raise TimeoutError("Fixture worker did not finish")
        self.detector.last_result = dict(
            local=self.hybrid.local.work.error if self.hybrid.local.work else None,
            remote=self.hybrid.remote.work.error if self.hybrid.remote.work else None,
        )

    def close(self):
        self.stack.close()
        for worker in (self.hybrid.local, self.hybrid.remote):
            worker._thread.join(1)
        self.backend.closed = self.locals.closed and self.remotes.closed
        self.closed = self.backend.closed


def run():
    cases = [
        run_case(p, b, t, f, rig_factory=HybridRig)
        for p, b, t, f in itertools.product(("remote", "hybrid"), BEHAVIOURS, (False, True), FAULTS)
    ]
    failed = sum(c["status"] != "PASS" for c in cases)
    return dict(
        schema=1,
        status="FAIL" if failed else "COMPONENT_MATRIX_PASS",
        scope="Real AsyncDetector/HybridDetector/HybridFrameDetector/SkeletonStack with synthetic sensor/backend inputs; no HTTP or live admission.",
        counts=dict(
            executed=len(cases), passed=len(cases) - failed, failed=failed, blocked_combinations=0
        ),
        cases=cases,
        blocked=[],
    )


def main():
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run()
    write_report(report, args.output, Path(__file__).resolve().parents[1])
    print(json.dumps(dict(status=report["status"], **report["counts"])))
    return int(bool(report["counts"]["failed"]))


if __name__ == "__main__":
    raise SystemExit(main())
