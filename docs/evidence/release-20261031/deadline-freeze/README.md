# Supported presentation release — 31 October 2026

RELEASE FREEZE READY — implementation can stop and report writing can begin

Production source tested: `9d4fc530eb3e03825885b45e0ea155e99dc18343`, branch `codex/release-candidate-20261031`. The final freeze commit additionally records this evidence; `production-inventory.json` binds the actual code/config. Freeze is for the supported presentation profile, not G3 baseline or full-road obstacle admission.

## Runtime correction and reuse

Measured overruns came from unbounded independent sensor waits, synchronous raw RGB-D writes and server renderer jitter. One retained RGB-D join alone took 120.166 ms. All failures remain in `all-attempts.json` and the local raw run directories. Optional inference/LiDAR joins now share an original-capture-relative cutoff at 60 ms, leaving 40 ms inside the unchanged 100 ms watchdog; raw RGB-D persistence runs after acknowledgement (or after failure requests a stop). Missing current LiDAR stays invalid/UNKNOWN/HOLD; no old frame is reused and no source requirement is removed. Stage clocks identify tick, ego, RGB-D join, bundle and each input. The watchdog still covers acquisition through RPC acknowledgement and can latch independently; packet TTL and mission remain unchanged.

Inspected PR #69 (`5942fab`) and Mohammad's standalone adapter. The current project already reuses their controller/adapter and scenario-runner foundations. PR #69's delayed-frame/asynchronous dispatch was not ported into the exact-frame acknowledged boundary. Existing documented renderer controls were reused: Low quality plus 640x360 spectator window. DX11 failed warm-up on the unmodified server; retained DX12 works. The CPU-inference attempt also failed and is not the selected profile.

Exact Windows server arguments: `-dx12 -windowed -ResX=640 -ResY=360 -quality-level=Low -nosound`. A necessary restart was performed after verifying no relevant actors and asynchronous world state. Sensor cameras remain 640x360/90 degrees and the qualified 128-channel LiDAR remains unchanged. CARLA client/server 0.9.16, Town10HD_Opt and exact OpenDRIVE SHA are recorded. Python 3.12.11, Torch 2.6.0+cu124, NumPy 2.5.2, Ultralytics 8.4.177, CUDA RTX 4070 Laptop. All three model hashes pass; this is not the historical package subset from another machine.

## Five consecutive acceptance runs

No operational attempt was discarded. Each run has separately discarded, braked startup/model warm-up; no whole acceptance run was excluded. The failed A/B/C/D sequences are retained and explicitly superseded after fixes. All final E runs use the same clean production source.

| Run | Attempted/acknowledged | Expiries | Median ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| E01 | 120/120 | 0 | 61.464 | 63.075 | 78.419 |
| E02 | 120/120 | 0 | 61.531 | 63.676 | 83.963 |
| E03 | 120/120 | 0 | 61.641 | 63.569 | 83.180 |
| E04 | 120/120 | 0 | 61.363 | 64.965 | 74.678 |
| E05 | 120/120 | 0 | 61.530 | 62.721 | 73.450 |

Timing starts before owned tick/acquisition and ends at successful server RPC acknowledgement. It does not measure a physical brake-settling interval or a subsequent actuation tick. Stage distributions and slowest stage per run are in `five-run-summary.json`; every clock is retained in CSV and JSONL. Overall 600/600 acknowledgements, zero watchdog faults, 485 current local detections accepted. **Zero current LiDAR reports were ready for fusion in these 600 cycles.** LiDAR capture and background processing run, but mandatory fusion correctly yields UNKNOWN/HOLD. This is not a claim of timely valid fused observations or driving admission. The earlier four-ready-report run remains historical, not substituted into this denominator.

## Demonstrated and cut capabilities

- Live parked RGB-D capture, current local car tracking, LiDAR sensor capture, exact-frame rejection, wayfinder guidance, governed HOLD and acknowledged full-brake packets; one route installation. Forced brake is never described as a target-induced stop.
- Separate frozen navigation component: one complete final-source run, 1016 samples, zero collision/lane events/false stops, mean absolute CTE 0.020446 m, cruise RMS error 0.111257 m/s, receipt-to-apply p95 6.692 ms. One run is not five-run S0 acceptance and excludes obstacle perception. Historical five-run component evidence remains separately scoped.
- `parked-sensors.png` uses actual frame 669 and matching depth/raw LiDAR; LiDAR visualization is offline postprocessing. `parked-recording.mp4` has 120 recorded frames at 10 fps, boxes only from accepted current local detections; local decoding verified. It is a recorded parked clip, not live video or physical stop proof.
- `wayfinder/` renders 24 actual final navigation poses, including one real paused sample; source CSV and hashes retained. This is recorded mini-map replay, not live HUD.
- Existing offline fusion, synthetic P1 parking, recorded lane disagreement, protocol/fault and weather-plan evidence remains valid in its original scope. Weather settings preflight verifies installed fields without changing weather or capturing a dataset.
- CUT from live release: lane/semantic HUD (deadline failures), autonomous RESUME (#40), full-road perception-driven motion/normal stop (#41 false positive), live timely valid LiDAR fusion, hybrid/cloud operation, live P1 and reverse P2. Standalone navigation has no obstacle-avoidance claim. No unsupported criterion is marked PASS.

## Startup

```bash
cd /home/alan/release-candidate-20261031
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/alan/issue40-runtime312/bin/python scripts/run_release.py observe --execute --target --host 172.28.176.1 --output runs/release/presentation-parked-01
# Separate navigation component, on a cleared owned scene:
PYTHONPATH=src /home/alan/issue40-runtime312/bin/python scripts/run_release.py navigation --execute --host 172.28.176.1 --output runs/release/presentation-nav-01
```

Choose fresh output directories. For the presentation, prepare startup/model warm-up before the clock and use the verified recordings if live startup fails. The default `demo` command is explicitly synthetic. No feature engineering after freeze except an emergency presentation-critical fix.

## Validation and handoff

1641 pytest passed, 1 skipped, 3 subtests; one third-party deprecation warning. Admission/all 19 negative probes, Ruff, AST boundary and 124 blocked-simulator core imports, generated notes, model hashes and diff checks pass. Strict Sphinx is recorded with final documentation. Existing network-none Docker real-model protocol smoke is reused: service/deployment/protocol sources were unchanged, so no new Docker or cloud-speed claim is made. Original 32 dirty files remain byte-identical and untouched.

All new commit subjects pass Conventional Commits; no published history was rewritten. Human review is recorded per the user's 31 October statement in the PR descriptions and existing AI journal; no additional review certificate is required. This does not turn missing technical criteria into PASS. #99 retains its specific normal-stop limitation. Rehearsals, second-laptop playback and private individual submission confirmations remain team actions.

The issue reconciliation lists each original open issue and its actual remaining criterion. #54's implementation/replay criteria are complete; report index/docs and timed-script criteria can close with the documentation PR. The source runtime PR #101 remains open for lane/S5/full-stack acceptance; #99/#116 remain Draft for their unresolved contracts. The documentation PR may merge independently without asserting those missing results.
