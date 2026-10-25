"""Conservative multi-source corridor fusion (E2-04).

Each sensor path (RGB-D tracks, LiDAR tracks, ...) reduces its own evidence to one
:class:`SourceCorridorReport`: a source ID, whether its observation is valid, its
nearest valid front-bumper gap in metres and the track identities that produced it.
:func:`fuse_corridor_reports` combines any number of reports into one
:class:`FusedCorridorReport` with these rules:

* **Nearer wins.** The fused gap is the minimum valid gap across sources. When two
  sources both report a gap, their disagreement is recorded, never averaged.
* **Loss of a source never clears the corridor.** If any configured source is
  invalid or missing, the fused observation is invalid and carries every reason.
  A valid source that sees nothing contributes no gap, not clearance.
* **No double counting.** Source IDs must be unique, every track identity must be
  namespaced by its own source, and identities cannot repeat across sources. One
  obstacle seen by two sensors yields one gap (the nearer), not two obstacles.
* **Clearance needs every source.** ``clearance_valid`` is true only when every
  source independently asserts it. A LiDAR source asserts it only from clear road
  bins seen from the bumper to at least its ``clear_start_m`` (not from the mere
  absence of clusters), and the RGB-D clearance rule must agree, so it remains
  binding.

Nothing here imports a simulator or a controller; the autonomy layer turns the
fused report into its ``CorridorObservation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from obstacle_perception.types import TrackIdentity


@dataclass(frozen=True)
class SourceCorridorReport:
    """One sensor path's corridor evidence for one frame.

    ``nearest_gap_m`` and ``visible_distance_m`` are metres along the route from
    the front bumper; ``None`` means no valid obstacle gap / no visibility claim.
    ``track_identities`` lists the source-namespaced identities behind the report.
    """

    source_id: str
    valid: bool
    nearest_gap_m: float | None = None
    visible_distance_m: float | None = None
    clearance_valid: bool = False
    track_identities: tuple[TrackIdentity, ...] = ()
    invalid_reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be nonempty")
        if type(self.valid) is not bool or type(self.clearance_valid) is not bool:
            raise ValueError("valid and clearance_valid must be bool")
        for name in ("nearest_gap_m", "visible_distance_m"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not math.isfinite(value)):
                raise ValueError(f"{name} must be finite or None")
        if not self.valid:
            if not self.invalid_reason:
                raise ValueError("invalid reports require a reason")
            if self.nearest_gap_m is not None or self.clearance_valid:
                raise ValueError("invalid reports cannot carry a gap or clearance")
        elif self.invalid_reason is not None:
            raise ValueError("valid reports cannot carry an invalid reason")
        identities = tuple(self.track_identities)
        for identity in identities:
            if not isinstance(identity, TrackIdentity):
                raise TypeError("track_identities must be TrackIdentity")
            if identity.source_id != self.source_id:
                raise ValueError("track identity is not namespaced by this source")
        if len(set(identities)) != len(identities):
            raise ValueError("duplicate track identity within one source")
        object.__setattr__(self, "track_identities", identities)


@dataclass(frozen=True)
class FusedCorridorReport:
    """Result of fusing source reports; see module docstring for the rules."""

    valid: bool
    obstacle_gap_m: float | None
    visible_distance_m: float | None
    clearance_valid: bool
    nearest_source_id: str | None
    source_gaps_m: tuple[tuple[str, float | None], ...]
    disagreement_m: float | None
    invalid_reasons: tuple[str, ...]
    track_identities: tuple[TrackIdentity, ...]

    def to_dict(self):
        return dict(
            valid=self.valid,
            obstacle_gap_m=self.obstacle_gap_m,
            visible_distance_m=self.visible_distance_m,
            clearance_valid=self.clearance_valid,
            nearest_source_id=self.nearest_source_id,
            source_gaps_m=dict(self.source_gaps_m),
            disagreement_m=self.disagreement_m,
            invalid_reasons=list(self.invalid_reasons),
            track_count=len(self.track_identities),
        )


def fuse_corridor_reports(reports, *, required_sources):
    """Combine per-source reports into one conservative corridor report.

    ``required_sources`` names every source that must be present and valid; a
    missing or invalid one makes the fused report invalid (reason recorded) and
    withholds any gap, so loss of a sensor is reported, never read as clear.
    Gaps and visibility are in metres; the fused gap is the minimum valid gap.
    """
    reports = tuple(reports)
    required = tuple(required_sources)
    if not required or len(set(required)) != len(required):
        raise ValueError("required_sources must be a nonempty set of distinct IDs")
    seen = set()
    identities = []
    for report in reports:
        if not isinstance(report, SourceCorridorReport):
            raise TypeError("reports must be SourceCorridorReport")
        if report.source_id in seen:
            raise ValueError(f"duplicate source report: {report.source_id}")
        seen.add(report.source_id)
        identities.extend(report.track_identities)
    if len(set(identities)) != len(identities):
        raise ValueError("track identity repeated across sources")
    by_source = {r.source_id: r for r in reports}
    reasons = []
    for source in required:
        report = by_source.get(source)
        if report is None:
            reasons.append(f"{source}:missing")
        elif not report.valid:
            reasons.append(f"{source}:{report.invalid_reason}")
    valid = not reasons
    gaps = tuple((r.source_id, r.nearest_gap_m) for r in reports)
    valid_gaps = [(g, s) for s, g in gaps if g is not None and by_source[s].valid]
    nearest = min(valid_gaps) if valid_gaps else None
    visibilities = [
        r.visible_distance_m for r in reports if r.valid and r.visible_distance_m is not None
    ]
    disagreement = (
        float(max(g for g, _ in valid_gaps) - min(g for g, _ in valid_gaps))
        if len(valid_gaps) >= 2
        else None
    )
    return FusedCorridorReport(
        valid,
        None if not valid or nearest is None else float(nearest[0]),
        None if not valid or not visibilities else float(min(visibilities)),
        bool(valid and reports and all(r.clearance_valid for r in reports)),
        None if not valid or nearest is None else nearest[1],
        gaps,
        disagreement,
        tuple(reasons),
        tuple(identities),
    )


__all__ = ["FusedCorridorReport", "SourceCorridorReport", "fuse_corridor_reports"]
