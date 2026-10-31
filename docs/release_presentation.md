> **31 October live update:** Final source `13ff182` is BLOCKED by a reproduced 100 ms watchdog deadline failure after 72 acknowledged parked cycles. CARLA is available. See [live final evidence](evidence/release-20261031/live-final-20261031/README.md). Earlier offline-only or CARLA-unavailable statements below are historical; no release freeze, full HUD, normal stop, or RESUME acceptance is granted. New AI-assisted fixes await human review.

# Eight-minute demonstration script — #73 proposal

This is a timed script, not a claim that a human rehearsal occurred. New source,
script acceptance and G4 freeze remain pending actual review/live evidence.
Prepare dependencies/models and generate the assets before the presentation;
do not spend the eight minutes installing packages or loading cold models.
Use the candidate SHA in the evidence map and a new output directory per command.

The six proposed speaking roles below use official owner handles. These are
planning assignments, not a claim that people confirmed attendance, rehearsed
or approved this script. The team must confirm them before presentation.

| Time | Proposed speaker / cue | Action and expected result |
|---|---|---|
| 0:00–0:35 | `skwo0237` — introduction | State selected source/config and supported scope. Full sensor-driven normal-stop acceptance and autonomous RESUME remain incomplete. |
| 0:35–1:50 | `at0941` — runtime / S1–S2 evidence status | `PYTHONPATH=src python scripts/run_release.py --output runs/presentation/demo-01`. Show 80 synthetic cycles, governed commands then 60 full-brake cycles through obstacle removal/expiry, one route installation. Say explicitly that live S1/S2 proof is unavailable; this synthetic STOP/HOLD trace does not substitute for it or demonstrate RESUME. |
| 1:50–2:40 | `RickyPurani` — LiDAR / weather evidence | Show prepared `run_release.py lidar` summary: synthetic max gap error 0.034831 m and zero depth-loss-clear frames. Separately cite #103 recorded sensor error 0.063779 m with its conditions. Weather mode has 27 planned cells, all NOT_RUN. |
| 2:40–3:30 | `Abdullah-Almutairi-uni` — wayfinder | Show prepared `run_release.py wayfinder` mini-maps/guidance log, legal route/next turn and paused ETA. No lane-changing/rerouting claim. |
| 3:30–4:20 | `Zhaolin-Huang` — parking | Show prepared `run_release.py parking` plots/summary: eight synthetic P1 cases. Explicitly distinguish harness-only evidence, unadmitted live P1 and disabled reverse. |
| 4:20–5:20 | `kkaaman` — perception / diagnostic HUD / inference | Show prepared `run_release.py hud` outputs with INVALID/DISAGREE preserved. Actual recorded lane defects remain. Optional remote Docker protocol smoke passes; cloud speedup, baseline latency and accuracy are not established. If later Tier 1 succeeds, label its recording parked/full-brake-only. |
| 5:20–6:00 | `skwo0237` — validation / limitations / handover | Show final regression, 19 negative probes, pinned identities and evidence map. Explain missing restart coverage/physical bound and unresolved detector false positive. Hand over at exactly 6:00. |
| 6:00–8:00 | All six — reserved Q&A | Random cross-module questions: one owner explains their module; another explains a boundary/failure mode. Keep answers evidence-backed. This is a planned slot, not a claim that human Q&A occurred. |

**Timing: 360 seconds demonstration + 120 seconds reserved Q&A = 480 seconds.**
The script can be shown with CARLA OFF using retained figures/logs. Required live
S1/S2 proof is unavailable under current #40/#41 limitations; that scope item and
G4 acceptance remain open. Do not relabel synthetic evidence as live proof.

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
   (#74), backup playback (#68/#71), actual review and submissions remain pending.
