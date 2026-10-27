# Nonblocking remote inference and local fallback (#57)

## Status and boundaries

E4-04/E4-05 client and arbitration implementation and E4-07 replay tooling are
provided as optional, simulator-independent components. They are **not wired
into the admitted baseline** and do not establish G1/G2 or live CARLA acceptance.
E4-09 deployment latency/accuracy comparison remains **NOT RUN** until identical
annotated replay, model weights, hardware notes and actual endpoints are supplied.
The inference server/session implementation in #56 (E4-06/E4-08) is absent from
this source snapshot. Its maintainer must implement/confirm the wire contract
below; a loopback mock is not a production server or LAN/cloud evidence.

## Usage in a tick-driven owner

Install the optional dependencies with `python -m pip install '.[remote,test]'`.
Use an independently owned local contract Detector (for example FrameDetector).
Do not share a stateful backend with another worker or the synchronous baseline.
Construct `RemoteDetector(endpoint, source_id=..., model_version=...)`, then
`HybridDetector(local_backend, remote_worker)`. Reset with the current episode,
call `submit(frame)` once per increasing source frame, and call `poll()` from the
owner's existing tick. Both workers are submitted concurrently. `poll()` never
waits. When `decision.stop_required` is true, invoke the owner's existing controlled
stop path; do not interpret `batch=None` as a clear road. Process only the returned
same-frame batch through the existing depth/freshness/behaviour safeguards.
This new asynchronous API intentionally does not implement the synchronous
Detector protocol; replacing a synchronous detect() call with submit() alone
would be incorrect. A full control-loop adapter requires a separate integration
review. Old frame results are never reused against newer depth.

The submission budget starts at submission, so the owner must independently
reject already stale captures. The workers' inference deadline is not a claim
about full capture-to-actuation latency. Defaults here are experimental component
settings and do not alter the frozen baseline configuration.

One daemon worker per backend permits at most one in-flight job and no waiting
backlog. If a backend hangs, no replacement worker is created, including after
reset. The other worker can still provide fallback. Reset invalidates old
results and the next accepted job resets backend state on its own worker. Close
returns immediately and asks the worker to close its backend when it can exit;
it cannot interrupt a permanently hung library call. HTTP socket timeout bounds
individual blocking I/O, not an adversarial trickle response's total wall time;
the independent deadline discards late results and the fixed slot bounds resource
use. No hard real-time guarantee or external actuator watchdog is introduced.

Local results are usable on the first poll observing a timely valid completion.
Remote is preferred when timely and enabled. Failure or an unobserved intervening
remote frame disables preference; recovery requires three consecutive distinct
valid remote frames. Repeated polls of one frame cannot advance recovery. Local
work remains independently scheduled even while remote is healthy. The local
backend should be warmed before operational use by the integration owner;
continuous scheduling here prevents it being cold solely because remote won.

## JPEG HTTP v1 contract

POST an explicitly configured HTTP(S) endpoint with JSON:

```json
{
  "schema_version": 1, "episode_id": "run-01", "frame_id": 1,
  "sim_time_s": 0.05, "image_width": 640, "image_height": 360,
  "source_id": "front-rgb", "model_version": "weights-sha256",
  "jpeg_base64": "..."
}
```

The response must contain every identity field above except `jpeg_base64`, with
exact matching values and JSON types, plus exactly these fields:

```json
{
  "detections": [
    {"class_name": "car", "confidence": 0.9, "xyxy": [10, 20, 40, 80], "track_id": null}
  ],
  "valid": true, "invalid_reason": null, "server_duration_s": 0.012
}
```

The server measures `server_duration_s` as total request processing duration,
including server JPEG decode and inference. Boxes use original input-image pixel
coordinates. Empty `detections` with `valid=true` is a known valid empty result.
Invalid observations require a nonempty reason and no detections. Unknown keys,
duplicate JSON keys, mismatched/out-of-order identities, invalid contract values
and oversized replies are rejected. No confidence, box or timing value is silently
clipped. Default response limit is 2 MB. Remote tracking lifecycle is not defined
by this detection-only protocol; send null track IDs unless a reviewed server
identity/session policy exists. Authentication/TLS provisioning and endpoint
access belong to the deployment owner; do not embed credentials in URLs.

## Reproduce tests (Windows PowerShell)

From repository root on a separate feature branch/worktree:

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_remote_hybrid_detector.py
```

Tests use synthetic images, fake local backends and real loopback HTTP. They
cover malformed/invalid/stale replies, response bounds, HTTP timeouts, occupied
slots, reset isolation, timely local fallback, dual failure, expiry and recovery.
They are correctness tests, not model/network deployment measurements.

## Replay latency sweep

Provide at least 200 RGB uint8 HxWx3 `.npy` frames in a directory. The tool hashes
every input, excludes configured warm-up calls, and uses the same resized input
sequence per mode/configuration. The local factory is a zero-argument callable
returning a contract Detector using the same model identity as the servers.
Record CPU/GPU, software versions, weights hash, endpoint machine and network
conditions in hardware notes. Select an explicit provenance label.

```powershell
$env:PYTHONPATH="$PWD\src"
.\.venv\Scripts\python.exe -m scripts.benchmark_remote_detector --frames C:\replay\rgb --output runs\issue57\latency-01 --local-factory your_package:make_detector --model-version YOUR_WEIGHTS_SHA256 --hardware-notes "client/server hardware, versions and network" --origin recorded-model-replay --lan-endpoint http://SERVER:8000/detect --sizes 640x360 960x540 --qualities 70 85 --warmup 5
```

The output directory must be new. Outputs: `latency.csv`, `summary.json` and
`latency.png`. Per configuration, summary includes p50/p95/max successful-call
latency, failure count, and deadline-miss rate with **all attempts** in its
denominator (errors count as misses). Inspect failures before comparing results.
Local timing excludes JPEG/network; remote total includes client encoding, HTTP,
validation and conversion. Local runs once per image size, since JPEG quality
does not apply. The pooled CDF is a visualization only; use per-configuration
summary rows for comparisons. Inputs are loaded/resized before the timed detector
call: this measures detector-call end-to-end latency, not sensor-to-actuator time.

`rtt_residual_s = HTTP round trip - server-reported duration` includes transfer,
queueing and clock/instrumentation uncertainty. It is not separate upload/download
timing and may be negative under inconsistent instrumentation; inspect raw values.
No one-way estimates are produced. Missing LAN/cloud endpoints are explicitly
`NOT RUN`. Supply `--annotations labels.json` for accuracy. The JSON maps every .npy filename
to a list of objects with `class_name` and original-resolution `xyxy` coordinates;
empty ground truth must be an explicit empty list. Matching uses class-aware,
confidence-ordered one-to-one IoU >= 0.5, reported as micro precision/recall.
Failed calls count as missing detections. The annotation file hash is retained.
The generated `comparison.md` places metrics side by side. Accuracy is
`NOT_MEASURED` without annotated evaluation; do not close
E4-09 based solely on this latency output. Store actual replay CSV/plots and
accuracy results with team-accessible provenance, then link them in #47/#62.

## Remaining acceptance work

- Confirm #56 server protocol/session/model identity and ready endpoints.
- Wire arbitration into an explicitly reviewed integration owner, including warm-up
  and controlled-stop dispatch; component tests do not certify actuator behaviour.
- Collect identical >=200-frame model replays per configuration on available
  local/LAN/cloud deployments; retain raw timing and comparable accuracy evidence.
- Review actual outputs and limitations before marking #57 complete.

AI assistance generated this implementation, tests and documentation from the
uploaded main snapshot. Validation uses synthetic/loopback data and must not be
represented as actual detector accuracy, cloud performance or live acceptance.

## G1 source-inventory impact

The admitted record binds **every** Python source file, including optional code.
Adding these four modules therefore makes the current source inventory differ
from the admitted baseline even without wiring them into control. The existing
G1 checker must reject this feature checkout until its exact inventory is
reviewed and rebound by the designated owner. Two current-record G1 regression
checks consequently fail on this draft. Do not edit the validator, exclude these
modules, rewrite historical records, or claim existing G1 admission covers this
change. Keep this work in its separate feature branch and coordinate the normal
review/rebinding process before merge/live use. No admission record was changed.
