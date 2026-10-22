"""Explicit startup entry point for freshness-gated tracked governance."""

from autonomy.freshness import OperationalFreshnessGate
from autonomy.stack import SkeletonStack, StackConfig
from perception.frame_detector import FrameDetector


def operational_stack(backend, policy, *, source_id, model_version, simulation_clock, config=None):
    """Reset before stepping; policy has evaluate/reset/on_hold and owns tracking.

    capture_started_monotonic_s must come from the capture owner before its tick,
    not from dequeuing or inference completion. Consumers must apply ControlPacket
    through both clock expiry checks, e.g. FixedVehicleAdapter.apply_packet.
    """
    config = config or StackConfig(require_depth=True)
    if not config.require_depth:
        raise ValueError("Operational obstacle governance requires aligned depth")
    detector = FrameDetector(backend, source_id=source_id, model_version=model_version)
    gate = OperationalFreshnessGate(config, required_results=policy.temporal_filter.policy.hits)
    return SkeletonStack(
        detector,
        config,
        policy,
        freshness_gate=gate,
        simulation_clock=simulation_clock,
        use_tracking=True,
    )
