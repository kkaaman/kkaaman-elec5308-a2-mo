"""Strict versioned runtime configuration and reproducible content identities."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from autonomy.baseline import BaselineConfig
from navigation_control.types import ControllerConfig
from obstacle_perception.track_filter import TrackFilterConfig
from planning.behaviour import BehaviourConfig
from planning.speed_governor import GovernorConfig


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def read_config(path):
    document = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if document.get("schema_version") != 1 or document.get("status") not in {
        "PROVISIONAL",
        "FROZEN_FOR_VALIDATION",
    }:
        raise ValueError("Unsupported configuration schema/status")
    values = dict(document["runtime"])
    for name, constructor in (
        ("controller", ControllerConfig),
        ("governor", GovernorConfig),
        ("behaviour", BehaviourConfig),
        ("tracks", TrackFilterConfig),
    ):
        values[name] = constructor(**values[name])
    config = BaselineConfig(**values)
    digest = content_hash(asdict(config))
    if document.get("runtime_sha256", digest) != digest:
        raise ValueError("Configuration hash mismatch")
    return config, document, digest


def write_config(path, config, *, status="PROVISIONAL", evidence=None):
    runtime = asdict(config)
    document = dict(
        schema_version=1,
        status=status,
        runtime=runtime,
        runtime_sha256=content_hash(runtime),
        evidence=evidence or {},
    )
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return document
