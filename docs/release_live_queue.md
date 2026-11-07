# Consolidated post-freeze acceptance queue — 1 November 2026

CARLA remains OFF until the offline handoff is complete. Use one already-running
CARLA 0.9.16 / Town10HD_Opt server with no other synchronous client. Do not change
the immutable presentation tag/worktree. Set PYTHONPATH=src:. in the new candidate
checkout, choose the existing service-capable Python environment, and use fresh
output paths. Existing accepted parked release evidence is reused.

## One stationary capture batch

| Issues | Site / repeats | PASS criterion | Canonical new evidence |
|---|---|---|---|
| #43/#49/#50; pristine input for #101/#42 | Existing mapped-road collector; 27 paired S0/S1/S2 × ClearNoon/HardRainNoon/ClearNight × 3 seeds; 600 frames/cell, 20 frames/scene | Every capture COLLECTED, identical map/rig, exact weather/light readbacks, no leaks/settings restored, validated held-out split; tables retain every failure. Detector quality and lane-centre correctness require review; no driving PASS inferred. | runs/postfreeze/paired plus derived TEST-09/weather/lane outputs |

```bash
python -m scripts.capture_weather_dataset --execute --host 127.0.0.1 --samples-per-cell 600 --output runs/postfreeze/paired
python scripts/prepare_weather_matrix.py --output runs/postfreeze/weather-matrix.json
python -m scripts.benchmark_perception --dataset runs/postfreeze/paired --depth-calibration docs/evidence/BL-C-04-06/depth-calibration.json --output runs/postfreeze/perception
python -m scripts.evaluate_captured_weather --dataset runs/postfreeze/paired --records runs/postfreeze/perception/records.jsonl --matrix runs/postfreeze/weather-matrix.json --output runs/postfreeze/weather
```

On the updated PR101 candidate, evaluate pristine lane-marking masks from those
same files (use the pinned lane SHA from configs/reproduction.json):

```bash
python scripts/evaluate_lane_dataset.py --dataset runs/postfreeze/paired --weights lane_unet_best.pth --weights-sha256 10dca01621fb8e110f1d5b8419e81fb616a8ce1b75d6aface11f5793af88e8ff --output runs/postfreeze/lane-masks.json
```

The model SHA above must match the canonical manifest; it is verified before
loading. Binary mask quality does not prove correct lane-divider selection.
Inspect preserved recorded-curve disagreements; do not tune on test scenes.

## Reuse the same held-out images for #57 (no CARLA needed afterward)

Provide an actual available LAN endpoint running the pinned model. Without that
hardware/service, LAN remains NOT RUN; cloud may remain unavailable. This step
does not need another simulator session.

```bash
python -m scripts.benchmark_remote_detector --dataset --frames runs/postfreeze/paired --output runs/postfreeze/inference-comparison --local-factory scripts.replay_local_detector:make_detector --model-version yolo26n:sha256:9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef --hardware-notes 'record actual client/server CPU/GPU, versions, LAN topology' --origin recorded-model-replay --lan-endpoint http://LAN_SERVER:8000/detect --sizes 640x360 960x540 --qualities 70 85 --warmup 5
```

Pass: >=200 same held-out images/configuration, model identity checks, all-attempt
deadline misses and successful-call p50/p95/max; per-class/visible-mask accuracy
scope and failures recorded. No one-way upload/download timing claim. The
LAN_SERVER hostname is the one externally supplied deployment input.

## Conditional moving acceptance — do not run yet

#40 positive full-region clearance/autonomous RESUME and #41 genuine normal
target-induced stopping remain unvalidated. #116 stays Draft/open; #99 keeps its
specific disclosure. No already-developed new solution was found to justify
reopening exhausted research.

#44/#47/#48/#51/#53/#55/#58/#60/#63/#67 require the relevant missing baseline or
extension admission. Their independent infrastructure/docs are retained; another
CARLA session cannot fix an unsupported contract. #42/#45/#46/#70 still require
applicable final full-stack S0/S3/S4/S5, faults/60-second HOLD, lane-divider and
held-green smoke evidence. Do not label S0 navigation-only or parked full-brake
runs as those results. There is no admitted final S1–S5 command; no fictitious
command is supplied. Once prerequisites are solved, collect 30 baseline trials
and compatible extension/fault workload evidence in the same continued session.

Use the existing scenario evaluators and attempt register; preserve failures.
Five reachable Wayfinder route pairs, live P1 tolerance/blocked-bay trials,
server-off failover and combined flags follow their actual admission
preconditions. Reverse P2 is cut. Physical backup-laptop cold start/playback,
rehearsal/Q&A and confidential submissions remain human actions.
