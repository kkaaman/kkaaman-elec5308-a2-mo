# Eight-minute demonstration script — #73

Supported implementation is frozen; [final measured evidence](evidence/release-20261031/deadline-freeze/README.md) supplies five live parked runs, an actual recorded sensor clip and a separate navigation-component run. This is a timed script, not a claim that a team rehearsal occurred. Prepare models/live startup and recordings before the clock. Human review is recorded per the user's 31 October statement; attendance and role confirmation remain team actions.

The six proposed speaking roles below use official owner handles. These are
planning assignments, not a claim that people confirmed attendance, rehearsed
or approved this script. The team must confirm them before presentation.

| Time | Proposed speaker / cue | Action and expected result |
|---|---|---|
| 0:00–0:35 | `skwo0237` — introduction | State selected source/config and supported scope. Full sensor-driven normal-stop acceptance and autonomous RESUME remain incomplete. |
| 0:35–1:50 | `at0941` — runtime / S1–S2 evidence status | Show the prepared `deadline-freeze/parked-recording.mp4` and matched sensor figure, or launch the exact parked command in the freeze record before the clock. State 600/600 acknowledged full-brake cycles across five runs; this is parked tracking/HOLD evidence, not live S1/S2 target-induced STOP/RESUME. |
| 1:50–2:40 | `RickyPurani` — LiDAR / weather evidence | Show prepared `run_release.py lidar` summary: synthetic max gap error 0.034831 m and zero depth-loss-clear frames. Separately cite #103 recorded sensor error 0.063779 m with its conditions. Weather mode has 27 planned cells, all NOT_RUN. |
| 2:40–3:30 | `Abdullah-Almutairi-uni` — wayfinder | Show `deadline-freeze/wayfinder/` mini-maps from actual final navigation poses, legal route/next turn and one real paused ETA. Mention the separate navigation-component run: full route, zero collision/lane events, 0.020446 m mean absolute CTE. No lane-changing/rerouting claim. |
| 3:30–4:20 | `Zhaolin-Huang` — parking | Show prepared `run_release.py parking` plots/summary: eight synthetic P1 cases. Explicitly distinguish harness-only evidence, unadmitted live P1 and disabled reverse. |
| 4:20–5:20 | `kkaaman` — perception / diagnostic HUD / inference | Show prepared `run_release.py hud` outputs with INVALID/DISAGREE preserved. Actual recorded lane defects remain. Optional remote Docker protocol smoke passes; cloud speedup, baseline latency and accuracy are not established. The actual parked clip is recorded/full-brake-only; live LiDAR fusion stayed unavailable and live HUD is cut. |
| 5:20–6:00 | `skwo0237` — validation / limitations / handover | Show final regression, 19 negative probes, pinned identities and evidence map. Explain missing restart coverage/physical bound and unresolved detector false positive. Hand over at exactly 6:00. |
| 6:00–8:00 | All six — reserved Q&A | Random cross-module questions: one owner explains their module; another explains a boundary/failure mode. Keep answers evidence-backed. This is a planned slot, not a claim that human Q&A occurred. |

**Timing: 360 seconds demonstration + 120 seconds reserved Q&A = 480 seconds.**
The script can be shown with CARLA OFF using retained figures/logs. Required live
S1/S2 proof is unavailable under current #40/#41 limitations; that scope item and
full-baseline acceptance remain open. Do not relabel synthetic evidence as live proof.

Prepare each asset before the clock with a new output directory:

```bash
PYTHONPATH=src python scripts/run_release.py lidar --output runs/presentation/lidar-01
PYTHONPATH=src python scripts/run_release.py wayfinder --output runs/presentation/wayfinder-01
PYTHONPATH=src python scripts/run_release.py parking --output runs/presentation/parking-01
PYTHONPATH=src python scripts/run_release.py hud --output runs/presentation/hud-01
PYTHONPATH=src python scripts/run_release.py weather --output runs/presentation/weather-01
```

## Operator checklist and recovery to recorded assets

1. Before the clock: verify source SHA, models (`scripts/check_reproduction.py
   --assets-only`), tested environment, local PNGs/logs and backup video playback
   on both laptops. Prepare the six offline modes above with unique directories.
2. Open the selected figures/logs and the existing recorded clip/overlay assets.
   Keep `docs/release_evidence_map.md` alongside every displayed claim.
3. If a command fails, keep its output/error. Switch within **15 seconds** to the
   pre-generated corresponding PNG/JSON. State “recorded/offline evidence” and
   its source conditions; never narrate it as a current live run.
4. If later CARLA demonstration faults, request keyboard interruption and verify
   actual full brake/cleanup. If actuator connectivity is lost, the operator
   must control the simulator; do not report an unacknowledged stop as success.
   Continue with the verified backup video or labelled frames/logs.
5. If no backup video has actually been selected/verified, use the static assets
   and disclose that limitation. Do not write “video recovery verified”.
6. Human rehearsal (#75/#76), display readability, cold start/keyboard-stop
   (#74), backup playback (#68/#71), submissions remain pending.
