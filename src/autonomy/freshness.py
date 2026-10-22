"""Operational freshness gate using existing stack and command contracts."""

import math


class OperationalFreshnessGate:
    def __init__(self, config, *, required_results):
        if type(required_results) is not int or required_results < 1:
            raise ValueError("Positive confirmation-derived warm-up count required")
        self.config = config
        self.required_results = required_results
        self.reset(episode_id=None, generation=None)

    def reset(self, *, episode_id, generation):
        self.episode_id, self.generation = episode_id, generation
        self.fresh_results = 0
        self.reason = "startup_hold"

    def reject(self, reason):
        self.fresh_results = 0
        self.reason = reason
        return False

    def check(self, frame, *, now, sim_time_s, generation, processing_s=0.0):
        if (frame.episode_id, generation) != (self.episode_id, self.generation):
            self.reset(episode_id=frame.episode_id, generation=generation)
            return self.reject("detector_reset_hold")
        origin = frame.capture_started_monotonic_s
        if origin is None or not all(math.isfinite(x) for x in (now, sim_time_s, processing_s)):
            return self.reject("missing_capture_clock")
        age = now - origin
        if age < 0 or age > self.config.max_observation_age_s:
            return self.reject("capture_age_hold")
        if (
            sim_time_s < frame.sim_time_s
            or sim_time_s >= frame.sim_time_s + self.config.command_ttl_s
        ):
            return self.reject("source_command_expired")
        if processing_s < 0 or processing_s > self.config.max_step_wall_s:
            return self.reject("processing_budget_hold")
        return True

    def result(self, frame, *, valid, **clocks):
        fresh = valid and self.check(frame, **clocks)
        if not fresh:
            if not valid:
                self.reject("invalid_perception_hold")
            return False, False
        self.fresh_results = min(self.required_results, self.fresh_results + 1)
        ready = self.fresh_results >= self.required_results
        self.reason = "operational" if ready else "warmup_hold"
        return True, ready
