> **31 October live update:** Final source `13ff182` is BLOCKED by a reproduced 100 ms watchdog deadline failure after 72 acknowledged parked cycles. CARLA is available. See [live final evidence](evidence/release-20261031/live-final-20261031/README.md). Earlier offline-only or CARLA-unavailable statements below are historical; no release freeze, full HUD, normal stop, or RESUME acceptance is granted. New AI-assisted fixes await human review.

# Consolidated CARLA validation queue — not executed

CARLA was not started, connected to or ticked during this release pass.
Use the exact candidate SHA recorded in the release evidence map. Human review
of the new source and admission inventory must occur before acceptance. No tag.
Matching CARLA **0.9.16**, **Town10HD_Opt**, synchronous **0.05 s**, empty world,
exclusive tick owner, rendered RGB-D, pinned local weights and validated
`configs/lidar_issue40.json` are required. Do not reload/start the simulator from
these commands. Replace the example WSL endpoint with the actual native host.

## Tier 1 — one parked composition session

First run offline preflight:

```bash
PYTHONPATH=src python scripts/run_release.py observe --output runs/release/live-preflight-01
```

Then, only in the later live session:

```bash
PYTHONPATH=src python scripts/run_release.py observe --execute --host 172.28.176.1 --output runs/release/live-headless-01
PYTHONPATH=src python scripts/run_release.py observe --execute --hud --host 172.28.176.1 --output runs/release/live-hud-01
```

Owners #41/#99, #42/#45/#101, #64/#67/#70/#74. Two collections, each 120
operational ticks after discarded warm-up, at the frozen straight-route site
(rear-axle arc 79 m; map/road membership checked). Both runs stay physically
parked with full brake. Do not substitute a spawn chosen to suppress the known
false positive. Retain every attempt, including watchdog/timeout failures.

Evidence: raw RGB-D NPZ, LiDAR raw bytes and records, exact calibration/source
manifest, capture/dispatch/acknowledgement JSONL, guidance, optional HUD PNGs and
summary. PASS for this bounded smoke means all requested cycles complete,
zero positive applied commands, actual speed <=0.1 m/s, current-frame required
sources, useful logs, no unexpected watchdog fault, resource cleanup and settings
restoration independently checked after exit. Compare actual capture-to-apply
p95 against the unchanged 100 ms gate, report both workloads and counts. Missing
HUD is safe baseline operation but fails the HUD demonstration criterion. Check
keyboard interruption once while parked; retain final brake/cleanup observations.

This proves integration/readiness only. No claim of route completion, target
stopping, lane accuracy, 60-second HOLD or autonomous RESUME follows from it.
Failure blocks the affected live presentation claim and source-freeze readiness,
not the already passing offline demonstration. Repair an ordinary integration
bug and rerun this bounded pair; never extend TTL to pass it.

## Tier 2 — one optional navigation diagnostic and backup selection

```bash
PYTHONPATH=src python scripts/run_scenario.py --runs 1 --host 172.28.176.1 --output runs/release/nav-01
```

Owners #46/#55/#67/#74. One frozen route, same map/vehicle/timestep, RGB capture
and measured rear-axle localization. This runner is navigation/controller-only;
it does not become full perception S0 because RGB is present. Check route reaches
goal, no collisions, one route installation and readable guidance/backup assets.
A full S0 acceptance refresh, if separately needed, uses the same command with
`--runs 5`; do not rerun those five merely to recreate the already retained
component result. Human presenters must verify backup playback on both laptops
and rehearse actual interruption/recovery. This blocks live navigation/backup
claims if unsuccessful; offline Wayfinder/P1 demonstrations remain available.

## Tier 3 — acceptance work deliberately not dispatched

| Issues / PR | Missing evidence and prerequisite | Repeats | Impact |
|---|---|---|---|
| #40/#44/#116 | Validated full restart coverage plus physical acceleration bound before any autonomous-resume trial | Original S1/S2 and repeated stop-margin criteria, only after premise changes | Resume/issue closure |
| #41/#99 | Reviewed semantic correction or a genuinely new validated capability before retrying the disproved normal-stop setup | One complete normal first stop, then required regression | Normal-stop claim/closure |
| #42/#45/#101 | Pristine lane labels/correction, S3/S4/S5 and target-induced stop runner admission | S5 five; continuous S4 hold and declared other repeats | Lane/acceptance claims |
| #47/#48/#67 | Supported baseline operating scope and all prerequisite acceptance | S0-S5 five each (30 total), failed runs retained | G3/G4 freeze |
| #49-#51 | Condition-labelled capture/held-out dataset and admitted scenarios | 27 paired weather cells, then required baseline regression | Weather claims |
| #53/#55/#60 | Separately admitted LiDAR driving, five legal live Wayfinder pairs, correct live parking sensor/actuation | L1 protocol; five destination pairs; five P1 trials; baseline regression as required | Driving/extension admission |
| #57/#58 | Intended LAN/cloud/hardware, >=200 held-out frames, actual failover trials | Held-out frame and failover/baseline protocols | Remote/hybrid claims |
| #61 | P2 remains cut; no automatic reverse pad session | None for selected release | Future-work label |
| #71/#75/#76/#78-#85 | Physical backup machine, actual people and private submissions | Human-owned | Rehearsal/course/submission |

These are **not commands ready to run**. No safe executable command is supplied
for an unsupported acceptance experiment. The exact missing official criteria
remain in the issue graph; a script invocation cannot supply their prerequisites.
Do not put #40 ray-cell research or #41 trash-can threshold tuning back in the
live queue. The smallest executable session is Tier 1, with Tier 2 only if the
presentation needs a new navigation recording. No automatic 30+27 replay batch.
