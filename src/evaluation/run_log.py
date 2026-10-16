"""Versioned JSONL run logs; host times use one relative monotonic origin."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path


def plain(value):
    if is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(plain(value), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance(root, model_paths=()):
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"git_hash": git("rev-parse", "HEAD"), "git_dirty": None if status is None else bool(status),
            "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"])),
            "dependency_metadata_sha256": {d.metadata["Name"]: hashlib.sha256((d.read_text("METADATA") or "").encode()).hexdigest() for d in importlib.metadata.distributions() if d.metadata["Name"]},
            "model_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in model_paths},
            "hardware": {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()},
            "python": platform.python_version()}


class RunLogger:
    """Append matched observations, commands and evaluation-only truth."""

    def __init__(self, directory, *, config, seed, metadata, host_origin_s):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.origin = host_origin_s
        self.last_frame, self.last_time = -1, -1.0
        write_json(self.directory / "manifest.json", {"schema_version": 2, "config": config,
                   "seed": seed, "provenance": metadata, "host_clock": "monotonic relative to run origin"})
        self.ticks = (self.directory / "ticks.jsonl").open("w", encoding="utf-8")
        self.events = (self.directory / "evaluation.jsonl").open("w", encoding="utf-8")

    def record(self, bundle, packet, diagnostics, evaluation, *, host_applied_monotonic_s,
               applied_sim_time_s, goal_distance_m, planned_final_stop=False):
        ids = (bundle.frame_id, packet.source_frame_id, diagnostics["frame_id"], evaluation["frame_id"])
        if len(set(ids)) != 1 or ids[0] <= self.last_frame:
            raise ValueError("frames must match and increase")
        if packet.episode_id != bundle.episode_id or packet.source_sim_time_s != bundle.sim_time_s:
            raise ValueError("command episode and simulation time must match observation")
        if bundle.sim_time_s <= self.last_time or diagnostics["sim_time_s"] != bundle.sim_time_s:
            raise ValueError("simulation times must match and increase")
        if not packet.is_current(bundle.episode_id, applied_sim_time_s):
            raise ValueError("command must be current when applied")
        if host_applied_monotonic_s < bundle.received_monotonic_s:
            raise ValueError("application timestamp must follow receive timestamp")
        if not diagnostics["reason"] or not packet.diagnostics.reasons:
            raise ValueError("commands must include decision reasons")
        row = {"frame_id": bundle.frame_id, "sim_time_s": bundle.sim_time_s,
               "host_receive_s": bundle.received_monotonic_s - self.origin,
               "host_applied_s": host_applied_monotonic_s - self.origin,
               "simulation_frame_age_s": applied_sim_time_s - bundle.sim_time_s,
               "ego": plain(bundle.ego.state), "control": plain(packet), "diagnostics": plain(diagnostics),
               "goal_distance_m": goal_distance_m, "planned_final_stop": planned_final_stop}
        tick = json.dumps(row, sort_keys=True, allow_nan=False)
        event = json.dumps(plain(evaluation), sort_keys=True, allow_nan=False)
        self.ticks.write(tick + "\n")
        self.events.write(event + "\n")
        self.ticks.flush()
        self.events.flush()
        self.last_frame, self.last_time = bundle.frame_id, bundle.sim_time_s

    def close(self):
        self.ticks.close()
        self.events.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
