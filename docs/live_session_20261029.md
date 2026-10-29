# CARLA live-session execution,29 October2026

The already-running CARLA0.9.16/Town10HD_Opt server was reused throughout.
No simulator restart or shutdown. WSL used native-host172.28.176.1:2000.
Final read confirmed asynchronous mode and no vehicles/walkers/sensors left.
Original dirty issue40 workspace preserved.

## #103: PASS — live LiDAR acceptance satisfied and merged

Merge511120957efa67707549cfc81c800472bd4f4c96;issue52 CLOSED.
[Full canonical evidence](evidence/E2-01-04/live-20261029/README.md).
All E2-01..04 criteria satisfied independently of40 clearance/driving admission.

Final matrix90/90 known-object checks,90/90 recorded corridor classifications:
10/20/30m ×0/+4.5/-4.5m ×10 samples,exclusive synchronous owner,dt0.05.
Two site22 complete matrices passed90/90 each. All360 attempted sweeps retained:
attempt01=30/90,attempt02=80/90,attempt03=90/90,attempt04=90/90.
Thus290 total checks passed/70 failed,including all failed sweeps/outliers.

Repaired evaluation road contamination in padded known-car search and a bounded
asynchronous destruction observation fence,without relaxing0.3m/minimum6point
criteria. Site10's additional roadside contamination remained FAIL; alternate
site22 passed identical60m straight protocol. No ad hoc crop erased that failure.

Final frame/timestamp/intensity/360degree/density checks pass:density0.8406875
of32000 configured rays.90 exact raw-sensor transforms checked. Maximum recorded
in-corridor gap error0.063779m. Raw/transformed/cropped points,ground classes,
bin outputs,initial/final cleanup snapshots and source snapshots preserved in
four CRC/member-SHA verified archives. Final settings restored/no owned leaks.
Supplemental [raw intensity check](evidence/live-session-20261029/raw-intensity-verification.json)
checks actual sensor bytes before clipping,not only normalized values.

1420 full/71 focused tests,admission19 negative probes,Ruff,boundary109,
generated notes,strict Sphinx,diff and archive/member checks pass.

## #40: BLOCKED — validated LiDAR still cannot satisfy current clearance

The actual roof rig's first road return is1.457093m beyond the measured bumper;
broad road-bin support starts1.5m. Across90 final sweeps there are zero road
returns in the first metre. Positive from-bumper clearance frames=0.
Descriptive clear_start_m8m permits a blind prefix/sparse gaps; it is not a
certificate for that prefix. Current source intentionally withholds positive
clearance. Empty/no-return bins stay UNKNOWN; ground filtering does not create
clear.90 stale and90 future checks reject.

The current rule is stricter than the older draft phrase that LiDAR asserts
clearance from clear bins: no continuous traversable prefix was established.
This does not impose a full-height contract or prove a different rig mandatory.
The missing premise is a validated sensor/operating-domain inference covering
the region actually traversed.

Preserved40 code and staged-resume review were inspected. No private LiDAR copy,
blind probing or unsupported integration; five valid clear observations AND0.4s
and track-expiry prohibition unchanged. S1/S3/S4/S5 resume20-run matrix NOT_RUN:
final admitted runner did not become genuinely executable.

## #101: partial live evidence; remains OPEN,not merged

[Exact official candidate/evidence](https://github.com/USYD-ELEC5308/26-S2-Assignment2-Group7/blob/d49fc698786033e4e3042e0942572df567a4e9ae/docs/evidence/RV-06-08-TEST-22/live-20261029/README.md)
is retained on its existing branch; source/feature not privately copied here.

Prescribed120s clip completed:573 indexed display frames/control samples,
zero collisions,clean cleanup. Historicalc6fb4bc source matches all62 pins;
no historical manifest weakened. Recorder endpoint/no-display options retain
data and honestly label diagnostic workload. Actual800×600/90degree camera
at2.4m and measured Audi offset1.256668m used,not guessed Model3/A2 geometry.

115 overlays/all573 rows retained. Valid nonjunction |CTE|>0.1m samples262;
agreement29/57 left,69/152 right,40/53 straight;mean|difference|1.152m,1.109m,
0.574m respectively. Model sometimes misses internal divider/selects outer road
markings. HUD keeps DISAGREE. Processed display/approximate command-pose matching
is not pristine frame-aligned accuracy; disagreement not labelled PASS or tuned
away. Further independent raw-image/label verification remains.

Frozen obstacle-free navigation S0:5/5 PASS,zero collisions/lane events/false
stops,worst mean|CTE|0.020446m,cruise RMS0.111257m/s,
capture-to-apply p9513.648009ms. This is not41 complete-perception acceptance.
Actual code hashes and dirty tree recorded; no clean final selected-release claim.

Parked2×120-frame warm shadow:COMPLETE,234 warm valid positive shadow decisions,
zero cold/stale positive outputs,p95 capture-to-decision34.234413ms,
receipt-to-decision20.094915ms,max45.135729ms. Ego physically braked.
The profile correctly remains BLOCKED as final timing: it ends before command
application. Different clock/workload windows are not compared as an improvement.

Remaining:S3/S4/S5 complete-stack runner/clearance unavailable,S5 5/5 real stops,
headless/full-HUD capture-to-apply on final candidate. No server restart to
manufacture a partial comparison. Final model/cadence integration gate blocked.
1463 full/50 focused tests,admission19 negatives,Ruff,boundary114,strict docs,
generated notes,diff and six archive/member checks pass. These validate tooling,
not missing live acceptance.42/45 OPEN.

## #41: no defensible sensor-derived correction; normal acceptance NOT_RUN

LiDAR genuine-car geometry does not establish semantic rejection of the recorded
trash-can false car. Conservative nearest-wins fusion retains RGB-D false
positives; no aligned false-object dataset/validated rejection contract exists.
All-track fail-closed semantics unchanged. #99 remains Draft/open. No normal
target-induced41 stop was run or claimed PASS.

## Other compatible queue work

Read-only weather preflight passed:installed settings verified,world_mutated=false,
dataset NOT_CAPTURED. No weather robustness claim or49 closure.
[Installed record](evidence/live-session-20261029/weather-installed-settings.json).

## Repository/admission state and exact next actions

Canonical main after103 is511120957efa67707549cfc81c800472bd4f4c96;this report is
documentation-only publication on top. Only52 closed this live session.
Open PRs99/101;other official issue numbers/titles retained in
[the state snapshot](evidence/live-session-20261029/official-state.json).
40,41,42,43,44,45,46,47,48,49,50,51,53,54,55,57,58,60,61,62,63 and downstream
release/report/human tasks remain open for their recorded technical/data gates.

Canonical loader-only fingerprint4bbcbe04975691ae8e98026611fd6be698426ba98729b3e3bfaf89aba2a566a9;
101 candidateb300c2c1dcb2c36804d5dac47838b7c4d036ceb4f20af9ed5659f55b94de1497.
Supported complete reviews/rebinds preserve cruise4,deceleration1.497566177392398,
response0.9500000052154064,margin6,depth25m,stationary-vehicle-only/deferred
pedestrians,model/rig/confirmation/freshness/all-track and historical runner.
19 source/config/model/scope/record mutation probes still reject.
CI failures were verified explicit billing annotations,no runner,zero steps;
executed regressions were run/fixed locally and not called billing failures.

Next technical action:establish a validated positive sensor/operating-domain
clearance rule for the traversed near-bumper region,then complete/admit40's
runner before the20-run matrix. No command exists for that final runner yet.
For101,validate divider/ego-lane interpretation on pristine frame-aligned RGB/
poses/labels and complete final workload integration before S3/S4/S5; no tuning
from the single displayed failure.41 stays blocked unless a defensible recorded
offline correction preserves genuine-car and all-track regressions first.

No further currently executable item in this queue remains. CARLA left running
for later authorized work. AI-assisted OpenAI Codex under explicit user delegation;
no invented human approval or skipped failed evidence.
