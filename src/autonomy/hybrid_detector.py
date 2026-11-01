"""Optional nonblocking hybrid detector adapter for the existing control stack.

The canonical SkeletonStack retains packet expiry, exception latching and HOLD
handling. This adapter never enables a remote path in the admitted baseline.
"""

from dataclasses import replace

import numpy as np

from autonomy.freshness import OperationalFreshnessGate
from autonomy.obstacle_policy import TrackedGovernancePolicy
from autonomy.stack import SkeletonStack, StackConfig
from obstacle_perception.tracking import TrackDepthFilter
from obstacle_perception.types import CallTiming, DetectionBatch, ObservationMetadata
from perception.async_detector import matches


class LocalTrackingBackend:
    """Expose FrameDetector.track to the existing bounded worker's detect contract."""

    def __init__(self, frame_detector):
        self.backend = frame_detector

    @property
    def source_id(self):
        return self.backend.source_id

    def reset(self, *, episode_id):
        self.backend.reset(episode_id=episode_id)

    def detect(self, frame):
        return self.backend.track(frame)

    def close(self):
        self.backend.close()


class HybridFrameDetector:
    """Submit early, then let SkeletonStack poll once without waiting on HTTP.

    Pending/failed/expired workers return an invalid observation; SkeletonStack
    consequently emits full brake through its existing policy HOLD path. Model
    mismatches raise and latch the stack's existing emergency state. A source
    switch rebinds only temporal identity/history, preserves the policy's HOLD
    latch and returns an invalid switch frame to require fresh reconfirmation.
    """

    def __init__(self, hybrid, policy, *, model_version, source_id="hybrid"):
        if not isinstance(policy, TrackedGovernancePolicy):
            raise TypeError("Use the existing tracked governance policy")
        ObservationMetadata("validation", 0, 0, source_id, model_version)
        self.hybrid, self.policy = hybrid, policy
        self.model_version, self.source_id = model_version, source_id
        self.generation = 0
        self.episode_id = None
        self.frame = None
        self.active_source = None
        self.last_decision = None
        self.worker_sources = {}

    def reset(self, *, episode_id):
        self.hybrid.reset(episode_id=episode_id)
        self.episode_id = episode_id
        self.generation += 1
        self.frame = self.active_source = self.last_decision = None
        self.worker_sources = {}

    def submit(self, frame):
        """Start both bounded workers; caller may perform other work before step()."""
        if frame.episode_id != self.episode_id:
            raise ValueError("Reset for the matching episode")
        if self.frame is not None and (frame.episode_id, frame.frame_id, frame.sim_time_s) == (
            self.frame.episode_id,
            self.frame.frame_id,
            self.frame.sim_time_s,
        ):
            if frame.calibration != self.frame.calibration or not np.array_equal(
                frame.rgb, self.frame.rgb
            ):
                raise ValueError("Reused frame identity with different RGB/calibration")
            return False, False
        accepted = self.hybrid.submit(frame)
        self.frame = frame
        return accepted

    def _unknown(self, frame, reason):
        return DetectionBatch(
            ObservationMetadata(
                frame.episode_id,
                frame.frame_id,
                frame.sim_time_s,
                self.source_id,
                self.model_version,
            ),
            (),
            frame.rgb.shape[1],
            frame.rgb.shape[0],
            CallTiming(0),
            False,
            reason,
        )

    def detect(self, frame):
        self.submit(frame)
        decision = self.hybrid.poll()
        remote_backend = getattr(getattr(self.hybrid, "remote", None), "backend", None)
        self.last_decision = dict(
            episode_id=frame.episode_id,
            frame_id=frame.frame_id,
            source=decision.source,
            reason=decision.reason,
            remote_session_id=getattr(remote_backend, "session_id", None),
            generation=self.generation,
        )
        if decision.stop_required or decision.batch is None:
            return self._unknown(frame, decision.reason)
        batch = decision.batch
        if decision.source not in ("local", "remote") or not matches(batch, frame):
            raise ValueError("Hybrid decision has invalid source/frame identity")
        # Validate the worker identity before rewriting it into a source/session
        # namespace. Otherwise an invalid source is laundered into trusted input.
        worker = getattr(self.hybrid, decision.source, None)
        expected = getattr(getattr(worker, "backend", None), "source_id", None)
        if expected is None:
            expected = self.worker_sources.setdefault(decision.source, batch.metadata.source_id)
        if batch.metadata.source_id != expected:
            raise ValueError("Hybrid worker source identity mismatch")
        if batch.metadata.model_version != self.model_version:
            raise ValueError("Hybrid model identity differs from configured weights")
        session = (
            getattr(remote_backend, "session_id", None) if decision.source == "remote" else None
        )
        namespace = f"{self.source_id}:{decision.source}" + (f":{session}" if session else "")
        batch = replace(batch, metadata=replace(batch.metadata, source_id=namespace))
        switched = self.active_source is not None and self.active_source != namespace
        if self.active_source != namespace:
            # Never policy.reset(): it would release the previous obstacle latch.
            self.policy.on_hold(frame)
            previous = self.policy.temporal_filter
            self.policy.temporal_filter = TrackDepthFilter(
                source_id=namespace,
                model_version=self.model_version,
                policy=previous.policy,
                depth_policy=previous.depth_policy,
            )
            self.policy.temporal_filter.reset(episode_id=self.episode_id)
            self.active_source = namespace
        if switched:
            self.last_decision["reason"] = "backend_switch_requires_reconfirmation"
            return self._unknown(frame, "backend_switch_requires_reconfirmation")
        return batch

    def track(self, frame):
        """Already-tracked worker output; no second inference or ID reassignment."""
        return self.detect(frame)

    def close(self):
        self.hybrid.close()


def hybrid_control_stack(hybrid, policy, *, model_version, simulation_clock, config=None):
    """Explicit extension factory reusing canonical warm-up, HOLD and expiry guards.

    Submit with ``stack.detector.submit(frame)`` as soon as an aligned frame is
    available; call ``stack.step(frame)`` at command time. Neither call waits for
    inference/HTTP. Late/missing work produces the existing full-brake packet.
    This is optional component integration, not frozen G2 entry-point admission.
    """
    config = config or StackConfig(require_depth=True)
    if not config.require_depth:
        raise ValueError("Tracked hybrid control requires aligned depth")
    detector = HybridFrameDetector(hybrid, policy, model_version=model_version)
    gate = OperationalFreshnessGate(config, required_results=policy.temporal_filter.policy.hits)
    return SkeletonStack(
        detector,
        config,
        policy,
        freshness_gate=gate,
        simulation_clock=simulation_clock,
        use_tracking=True,
    )
