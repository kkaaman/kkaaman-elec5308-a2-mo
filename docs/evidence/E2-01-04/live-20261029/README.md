# PR103 live acceptance,29 October2026

**E2-01..04 PASS**, separate from positive-clearance/driving admission.
CARLA0.9.16 client/server, Town10HD_Opt,exclusive synchronous owner,dt0.05.
Same capture script and90-sweep matrix:10/20/30m ×0/+4.5/-4.5m ×10 samples.

## Every attempt retained

| Attempt | Straight site | Sweeps | Known-object PASS/FAIL | Outcome |
|---|---|---:|---:|---|
| 01 | 10 |90|30/60|INCOMPLETE; road in padded evaluation search selected as target face; immediate destruction snapshot stale |
| 02 | 10 |90|80/10|INCOMPLETE; separate roadside return contaminates20m/right search; bounded cleanup fence confirms restoration |
| 03 | 22 |90|90/0|COMPLETE; same60m straight-site check,alternate unobstructed fixture |
| 04 | 22 |90|90/0|COMPLETE; final receipt/read-clock and exact raw-byte retention enabled |

No failed/outlier sweep discarded. Archives and per-member SHA manifests retain
all360 transformed sweeps, failures, script snapshots, replay and cleanup.
Attempts02–04 additionally retain exact sensor-frame float32 bytes.
Attempt04 includes transformed/cropped arrays,ground classes/heights,road and
occupied counts and UNKNOWN/OCCUPIED bin states for every sweep. Clear bins=0.
Attempt-register preserves initial and post-fence cleanup observations.
Archives pass CRC and member hashes. Original incomplete outcomes remain intact.

## Runner repairs without weaker acceptance

Known-car evaluation now excludes points below0.15m above the known box base,
so road inside the padded search cannot masquerade as the vehicle face. This
truth-derived crop is evaluation-only; it never feeds production obstacle/
clearance policy.0.3m face tolerance and minimum6 points unchanged. Regressions
reject road-only and displaced target data. Site10's additional roadside
contamination remained FAIL; no threshold was widened to erase it.
Site22 passed the same60m straight/non-junction/yaw checks via explicit site
selection in the existing runner. Baseline environment was not relabelled.

Cleanup now observes bounded asynchronous destruction completion after owner
restoration. It never destroys unrelated actors or accepts a remaining leak/
timeout. Raw-byte storage is optional and bounded by queue capacity. Default
operating path unchanged; receipt/read host clocks are retained separately from
simulation time. After final capture only an unused script local was removed;
the exact executed capture-script snapshot remains archived.

## Actual recorded replay and critical clearance audit

Final90/90 sweeps valid;90/90 target corridor/roadside classifications match;
maximum absolute in-corridor gap error0.063779m. All intensities checked in[0,1],
frame/tick timestamps match, scan azimuth360degrees,density0.8406875 of32000
configured rays. All90 raw sensor clouds transform exactly to saved ego points.

RGB-D was deliberately absent:90/90 fused HOLD,zero depth-loss frames treated
clear. Initial unconfirmed LiDAR tracks stay invalid; nearest/disagreement and
source-namespace fixtures also pass. No simultaneous RGB-D/LiDAR capture claimed.

The first road return is1.457093m ahead of measured front bumper; broad three-
strip road support starts at1.5m. Across90 sweeps,zero road returns in[0,1)m.
Observed ground elsewhere is not a continuous traversability certificate.
Empty/no-return bins remain UNKNOWN; ground filtering never generates clear.
Current source deliberately withholds positive clearance, including when the
descriptive road extent reaches beyond clear_start_m8m. That number allows an
unseen prefix and sparse rings; it is not evidence for that prefix.90 stale and
90 future-frame checks reject; maximum host receipt-to-read0.009311s.

Therefore #40 remains blocked on supported positive action/clearance bounds,
not an invented full-height or different-rig requirement. Five clear observations
AND0.4s and track-expiry prohibition remain unchanged. No resume run was made.
The genuine-car dataset provides no validated semantic rejection of #41's
trash-can false positive; normal #41 acceptance was not rerun.

## Validation and reproduction

71 focused LiDAR/replay/fusion tests;1420 full tests passed,1 CARLA deselected,
3 subtests,one dependency deprecation warning. Production admission and19
negative probes pass. Full production diff reviewed and supported loader-only
rebind recorded; frozen values/scope/model/sensors/historical runner unchanged.
Ruff,boundary109 modules,generated notes,strict Sphinx,diff checks also required.

Extract an attempt archive to a new directory,then:
python scripts/replay_lidar_obstacles.py --input EXTRACTED/sweeps.json --output NEW/replay.json
python scripts/audit_lidar_live.py --input EXTRACTED_ATTEMPT04/sweeps.json --output NEW/processed
Keep archive/member manifests and failed attempts. Actor truth only sets/evaluates
the fixture. AI-assisted OpenAI Codex continuation under explicit user delegation;
no new human approval claimed. This validates the issue52 obstacle path,not
extension53 performance/operational admission or #40/#41 safety acceptance.
