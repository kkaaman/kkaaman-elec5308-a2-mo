# Isolated inference service — official issue #56

The service implements the existing #105 JPEG/JSON client contract, including
run/session echo headers, with one independent ObjectDetector/tracker per session.
It is opt-in; no service or network path is enabled in the admitted baseline.

## Protocol and bounded resources

`POST /detect` takes JSON v1 fields: episode/frame/simulation-time identity,
image width/height, source/model identity and base64 JPEG. Required headers are
`X-Run-Id`, `X-Session-Id`, `X-Session-Start` and `X-Send-Monotonic-S`.
`X-Capture-Monotonic-S` is optional when actual capture start is available.
The first request explicitly starts a session; subsequent requests cannot silently
recreate expired or failed tracker state. Run/session values are echoed on
responses and checked by the existing client. Episode/source/model changes,
duplicate/decreasing frame IDs and non-increasing simulation time are rejected.

All client absolute clocks stay in the client domain. Capture must precede send;
neither is subtracted from a server clock. Server duration covers service
validation, session setup and inference on the server's own monotonic clock.
RTT residual remains transfer/queue/instrumentation uncertainty, not one-way time.

Authentication is `Authorization: Bearer <DETECTOR_TOKEN>` from environment.
`GET /health` reports configured model identity and explicitly does not certify
inference. `POST /reset` closes only the identified idle session. Active-session
contention/reset returns 409 without queuing. Missing/expired sessions return 410;
capacity returns 429, malformed input 422, oversized body 413 and unavailable
backend 503. A failed inference consumes its frame and poisons that session until
explicit reset. Errors do not expose backend exception details.

Defaults bound sessions to two, idle TTL to 60 server seconds, encoded JPEG to
4 MB, decoded pixels to 1920×1080 and JSON body to 6 MB. Uvicorn runs one worker
with at most eight concurrent connections. Multiple workers would each own
independent trackers and are unsupported. Each active session owns a model copy;
size CPU/GPU memory accordingly. Request deadlines and local fallback remain
client responsibilities; a service cannot authorize vehicle motion or clearance.

## Reproduce the CPU image

The complete 50-wheel hash lock targets Linux x86_64 / CPython 3.12. It is a
service-specific CPU environment, not the missing Windows/CUDA project-wide lock
from #62. The base image is digest-pinned. Preparation/build may use network for
base/OS packages and wheel acquisition; Python installation consumes only the
hash-checked local wheelhouse. First inference needs no network or model download.

```bash
python -m pip download --only-binary=:all: --require-hashes \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r deploy/inference/requirements-cpu.lock -d wheelhouse
docker build -f deploy/inference/Dockerfile -t group7-inference:issue56 .
```

Copy the built image (`docker save`/`docker load`), lock/manifest and local model
to a disconnected machine rather than expecting an offline rebuild to acquire a
base image or OS packages. Never commit credentials or large wheel/model copies.

Set `DETECTOR_TOKEN` to an explicit local secret and `DETECTOR_MODEL_SHA256` to
the exact mounted weights hash. The accepted model used in this smoke has SHA256
`9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
From a Linux shell with those environment variables exported:

```bash
docker run -d --name group7-inference --network none --read-only \
  --tmpfs /tmp:rw,nosuid,noexec,size=256m --memory 4g --cpus 2 --pids-limit 128 \
  --cap-drop ALL --security-opt no-new-privileges \
  --mount type=bind,src="$PWD/yolo26n.pt",dst=/models/yolo26n.pt,readonly \
  -e DETECTOR_TOKEN -e DETECTOR_MODEL_SHA256 \
  -e DETECTOR_WEIGHTS=/models/yolo26n.pt group7-inference:issue56
# Execute the real-client protocol smoke within the isolated network namespace:
docker exec -i group7-inference python - < scripts/smoke_inference_service.py
docker stop group7-inference
docker rm group7-inference
```

The network-none smoke checks loopback inside the container. For an actual LAN
deployment, use an explicitly selected private network/host binding and supply
its endpoint/token to the existing #105 client. LAN/cloud timing, accuracy and
control-runner integration remain #57/#58 work, not results of this smoke.

## Evidence and remaining admission

[Retained evidence](evidence/E4-01-03-06/phase-j-20261028/README.md) includes the
real-model image smoke, resource/source/image identities and build log. Fake
tracker tests exercise two clients, repeated episodes, TTL, contention, reset,
malformed JPEG/size/metadata, inference failure and client response identity.
A real-client/FastAPI fixture verifies repeated-episode cleanup at capacity one
and preserves the existing ClearanceGuard latch through reset/ID reuse. That is
an offline safety-contract regression, not live actuator certification.

This branch depends on the existing #105 client branch. Neither branch is merged
or operationally admitted: the exact G1 inventory changes and requires designated
review/rebinding. No trusted record, frozen threshold or safety guard is bypassed.
AI-assisted with OpenAI Codex, 28 October 2026; no human sign-off claimed.

```{toctree}
:hidden:

evidence/E4-01-03-06/phase-j-20261028/README
```
