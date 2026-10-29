# #40 LiDAR geometry investigation — 29 October 2026

**CARLA COMPLETE — sensor geometry validated; #40 offline integration remains**

Base: `8d9a797c92075a10b29fb00f65d4aab6f124db7b`. CARLA 0.9.16 was already
running; no restart was needed. All captures were parked, on Town10HD_Opt,
0.05 s synchronous ticks, with owned actors removed and previous settings
restored after each batch. No driving or #40 acceptance is implied.

## Original geometry and the 1.457 m result

#103 supplied the canonical capture, single transform, ground/clustering/tracking
and fusion path. Its original rig was actor-relative (0,0,2.4) m, zero
roll/pitch/yaw, 32 channels, −30° lower/+10° upper, 360° horizontal, 50 m range,
640000 points/s and 20 Hz rotation (32000 nominal returns/tick). Noise/drop-off
were zero. There is no minimum-range rejection: only nonfinite and >50 m
returns are dropped by decoding. Core processing crops at 40 m.

Measured Tesla geometry: actor→rear offset 1.386405669 m; front bumper
2.425108969 m ahead of actor / 3.811514638 m ahead of rear axle; half width
1.081725001 m; body vertical interval −0.007969439..1.479690611 m relative to
actor. The original first corridor road return was rear-frame
(5.268607616, −1.480898380, 0.001069450) m, hence **1.457092978 m beyond bumper**.
The raw lowest elevation measured approximately −30.0002°, including the
configured lower endpoint.

For a horizontal scanner at height h above the road, lowest-ray ground radius
is r=h/tan(30°). At lateral coordinate y, bumper gap is
`x_mount + sqrt(r² − y²) − x_bumper`. With h≈2.399 m and y≈−1.481 m,
this predicts ≈1.457 m. At y=0 the blind distance is ≈1.73 m. The earlier
1.457 m observation was off-axis, not a centreline range. Repeated roof
screening reproduces 1.457092978 m. This is physical mounting/FOV geometry,
not minimum-range filtering.

Filtering is separate: canonical EgoBody removes the ego box plus 0.3 m,
including a forward 0.3 m margin. Ground heights ROAD [−0.4,0.10] m,
RAISED (0.10,0.30) m and OBSTACLE [0.30,2.4) m retain their thresholds.
Road estimation uses 2 m bands, 3 m road window, 0.05 m surface depth,
tolerance 0.07+0.015/distance, maximum grade 0.12 and maximum band gap 4 m.
Cluster cells are 0.5 m, merge 1.5 m, minimum 6 points. Coverage-description
bins are 0.5 m longitudinal × three 1 m lateral strips over ±1.5 m, minimum
2 road points/strip. Existing `clear_start_m=8`, gap allowance 1+0.3×distance
and minimum-two-strips heuristic are unchanged and **are not certificates**.

## Actual region and what the observations establish

Using stopped speed ≤0.1 m/s, existing target ramp 1 m/s², tick 0.05 s,
response 0.9500000052154064 s, deceleration 1.497566177392398 m/s² and margin
6 m gives the conditional next-target envelope v=0.15 m/s:
`6 + v×response + v²/(2×deceleration) = 6.150012 m`.
At exact zero the first target is 0.05 m/s and the analogous distance is
6.048335 m. The target ramp is **not a hard physical acceleration bound**;
these numbers define a conservative screening region, not motion authority.
A production restart envelope must validate actual response/displacement.

Screen full bins from bumper through **6.5 m**, all three lateral strips across
**−1.5..+1.5 m**. The measured body is 2.16345 m wide, leaving ≈0.418 m each
side. Collision-relevant height spans road/contact through measured ego top
≈1.48 m, with terrain/body uncertainty still requiring a supported contract.
No demand for an unrelated full-height bounding volume or duplicate evidence
from both sensors has been introduced. LiDAR may eventually certify its
observed near region; RGB-D may supply farther regions where it has support.
The camera's near-ground blind prefix is ≈2.64 m, not a substitute for LiDAR.

Two observed ground returns per cell establish sampled **road support**, not
continuous traversability. Ego inflation can also remove the first 0.3 m even
when the remaining part of the first 0.5 m bin has points. Exact measured ego
masking is an evaluation comparison only: production mask/thresholds were not
weakened. Tiny measured first-return gaps are sampling distances, not claimed
submillimetre physical accuracy or proof of every point between rays.

## Screening and final choice

All candidate results, including failures, are retained. Initial screen:
8 candidates × 4 target distances (30/10/20/5 m) × 2 frames = 64 sweeps.
The table uses recomputation on full 0.5 m bins through 6.5 m and the measured
noninflated ego box; numbers are road-supported cells with target at 30 m.

| Candidate | Mount x,z m; pitch | Channels; pps; lower FOV | Support /39 |
|---|---|---|---:|
| Original roof | 0,2.4; 0° | 32;640k;−30° |31|
| Wider roof | 0,2.4; 0° |32;640k;−45°|34|
| Pitched roof |0,2.4;−15°|32;640k;−30°|36|
| Front low64 |2.5,0.9;0°|64;640k;−85°|36|
| Front mid64 |2.5,1.3;0°|64;640k;−85°|36|
| Front low128 |2.5,0.9;0°|128;1280k;−85°|39|
| Front mid128 |2.5,1.3;0°|128;1280k;−85°|39|
| **Front high128, selected** |**2.5,1.5;0°**|**128;1280k;−85°**|**39**|

All front candidates preserved the representative 10/20/30 m known-object
checks (6/6 each). The down-pitched roof lost the 30 m check. Roof ray/ego-AABB
intersection diagnostics indicate possible self occlusion; front candidates
had zero such intersections for the checked near-road rays. Parent/self returns
are present and must be masked, never assumed absent.

The first screen clipped the last bin at 6.15 m, initially favouring 1.5 m.
The full-bin audit corrects that endpoint artefact without deleting the original
results: all 128-channel front heights pass. A second screen checked all three
heights at pitch −1/0/+1° (36 sweeps): all 39 cells and all 36 object checks
passed. The selected 1.5 m mount is above the measured vehicle top, 7.49 cm
forward of its bumper; it avoids the low mount's larger far-ground angular
spacing while retaining no-pitch forward coverage. It is a plausible external
front sensor, not inside the road/body. No claim that 1.5 m uniquely passes or
is the global minimum is made. The standard 64→128 channel step resolved the
screened missing full cells without tuning classifier/bin thresholds.
Second-site check (spawn66, 8 sweeps) passed 39/39 cells and 8/8 object checks.
This validates the stated parked/straight/simulation scope, not arbitrary
weather, slope, bends or physical hardware performance.

Final configuration is protected in `configs/lidar_issue40.json` and matches
`LidarRig` defaults: **(2.5,0,1.5) m; roll/pitch/yaw0; 128 channels; 1.28M
points/s; −85..+10°; horizontal360°; range50m; rotation20Hz at dt0.05s**.
#103 remains implementation provenance. Pitch/roll configuration is now
supported; the single rigid transform was checked against CARLA's reflected
rotation matrix, maximum numerical discrepancy <8e−8 in three compound poses.

The known-object evaluator's padded 5 m search overlapped the ego and initially
selected its return. Optional measured ego-AABB exclusion fixes this evaluation
only; the .3 m tolerance and minimum 6 target points are unchanged. Original
screen failures are preserved. No truth box is supplied to production policy.

## Final qualification and fail-closed checks

* Existing matrix: 10/20/30 m × centre/+4.5/−4.5 m ×10 = **90/90 known-object
  checks and 90/90 classifications**, valid sweeps90/90; maximum gap error
  0.033880 m. All 90 have sampled road support in all 39 required cells.
* Near matrix: 5/5.5/6 m × centre/+0.5/−0.5 m ×10 = **90/90 known-object
  checks and 90/90 classifications**, valid90/90; maximum gap error0.091861 m.
  All90 show occupied cells and fail complete-road support. Startup unconfirmed
  tracks remain conservative (3 existing-matrix,9 near-matrix frames).
* Exact raw-byte hashes and decoded-array equality pass **180/180**; stale and
  future-frame report rejection pass **180/180 each**. Missing RGB-D causes
  **180/180 HOLD**, zero depth-loss frames clear. No capture callback errors.
* First measured physical road support beyond bumper: **0.000188805 m** at
  site22, **0.000189060 m** at site66 with exact ego mask. Canonical inflated
  mask first retained road point is **0.000496604 m**, located outside the
  laterally inflated body; this does not prove centreline sub-bin coverage.
* Offline extract-obstacles processing p95 is 10.73 ms existing /11.64 ms near;
  maxima15.12/13.13 ms. Callback-receipt→read maxima14.51/17.58 ms. These parked
  CPU component measurements are not end-to-end driving/operational admission.
* All five batches restored settings and leaked no owned actor after bounded
  cleanup verification. CARLA was never restarted.

A critical counterexample: centre target10 m has its nearest face inside the
required 6.5 m prefix; all road cells still contain support, **and occupied
returns coexist**. Road support alone cannot authorize RESUME. Empty/sparse,
invalid, stale, missing, expired or disappeared tracks remain no-clear.

## #40 and #41 remaining integration

The physical near-field problem is solvable. Canonical `LidarCorridorSource`
still deliberately returns `clear=False`; source fusion currently requires
both sources and has no regional certificate union. The preserved #40 work
remains untouched on `codex/issue40-stop-hold-resume` at ae6a83d plus its
uncommitted files. Its DepthCoverage has `axial_error_bound_m=None` and no
supported world-height bounds; its projected-full-box sufficient criterion
is an implementation assumption, not an issue requirement. It is not being
made mandatory or used to reject this sensor configuration.

Concrete offline work remains: define and test the stationary-vehicle observed
free-space/occlusion contract over the **actually traversed** region, regional
LiDAR/RGB-D composition with occupied/UNKNOWN veto and source binding, validate
the physical restart envelope, then wire the preserved StopResumeTrial into
the admitted runner. The generic RGB-D route-surface provider is not on main
(#99 remains Draft); silently copying that branch would bypass review. No
arbitrary depth error bound, obstacle minimum size or ground extrapolation was
invented to force a pass. Five clear observations **AND** ≥0.4 s, invalid/stale
HOLD, and no track-expiry clearance remain requirements.

Runner status: preserved recorder/removal lifecycle exists; canonical integrated
positive-clear runner is not complete/admitted. **S1/S3/S4/S5 ×5 NOT_RUN**:
no supported production clearance yet. Stop margin, steady HOLD/removal,
first clear observation, clear duration, RESUME delay and CRUISE re-entry are
therefore NOT_MEASURED, not passes. No frozen value or all-track guard changed.

#41: genuine cars remain geometrically supported. An occupied LiDAR cluster is
not semantic proof of a car and cannot reject the known trash-can false car.
There is no aligned false-positive-region capture with this new rig or validated
generic rejection rule. No special case, class filter, all-track relaxation or
#41 acceptance/closure is claimed. Current nearest-wins fusion retains the
original false positive; #41 remains open.

## Evidence and admission

[Evidence directory](evidence/issue40-geometry-20261029/README.md) retains all
288 new sweeps:64 initial+36 tolerance+8 second-site+90 qualification+90 near.
Each tar.xz has a SHA256 member manifest and was reopened to verify every member.
Source snapshots are retained inside capture/screen archives. The full-bin
recomputation is in `full-bin-audit.json`; rerun after extracting archives under
one directory with `PYTHONPATH=src:. python scripts/audit_issue40_geometry.py
--root EXTRACTED --output audit.json`.

Loader-only source identity: `ed42f70670961ea0792e8c637225ffa33dd15f15bbc6d7c0ba7484125ecf43fb`.
Mission file/semantic identities, model bytes, original runner binding and all
frozen values are unchanged; [review](evidence/issue40-geometry-20261029/admission-review.json)
records the intentional LiDAR source/config delta. Loader admission does not
admit a new live runner or make #40 complete. All19 negative probes rejected.

Validation logs in the evidence directory include 143 focused LiDAR/behaviour/
scenario tests, 54 separate preserved-workspace #40 tests, admission/19 negative
probes, full pytest (1428 passed,1 skipped,3 subtests), Ruff, simulator-boundary AST and 109 blocked-import smoke
imports, strict Sphinx and archive/member checks. The preserved-workspace tests
do not claim canonical runner integration. A capture CLI default-height mismatch
found during review was fixed and covered by a focused parser test.
