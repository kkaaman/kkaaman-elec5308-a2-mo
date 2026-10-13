# CARLA machines (PROC-02)

Every member with CARLA runs the same smoke test and records the result
here. The table decides the primary and backup demo machines, and keeps
version mismatches visible.

## Running the smoke test

Start the CARLA 0.9.16 server, close anything else that ticks the world,
then from the repository root:

```bash
python scripts/carla_smoke_test.py --member <your name>                  # 0.10 s step, 800×450
python scripts/carla_smoke_test.py --member <your name> --dt 0.05        # 20 Hz candidate
python scripts/carla_smoke_test.py --member <your name> --width 1280 --height 720
```

**What it checks:**
- client and server versions match;
- Town10HD loads;
- synchronous mode with a fixed step is applied;
- co-located RGB, depth, semantic and instance cameras deliver the **exact
  ticked frame** every tick;
- depth decodes to plausible metres;
- spawn and cleanup leave no actors behind, over two cycles;
- the original world settings are restored.

Results go to `runs/smoke/<host>-<time>/smoke.json`. Copy that JSON into
`docs/evidence/PROC-02/` and add your row below.

## Inventory

| Member | Machine | OS | CPU / RAM | GPU (VRAM, driver) | Python (CARLA env) | CARLA client / server | Smoke test |
|---|---|---|---|---|---|---|---|
| Mohammad | WCB3LJ64 (university desktop) | Windows 11 (10.0.26200) | i7-14700 / 31.7 GB | RTX 4000 SFF Ada (20 GB, 596.71) | 3.12.0 | 0.9.16 / 0.9.16 | **PASS**, 12 Oct 2026 |
| Abdullah | – | – | – | – | – | – | NOT RUN |
| Alan | – | – | – | – | – | – | NOT RUN |
| Zhaolin | – | – | – | – | – | – | NOT RUN |

## Results: Mohammad, WCB3LJ64

Vehicle `vehicle.lincoln.mkz_2020`, spawn index 0, Town10HD_Opt, 90° FOV,
100 ticks per cycle with the first 10 excluded from timing, and 2 cycles.
Five runs on 12 Oct 2026. All attempts are kept in
`docs/evidence/PROC-02/`.

| Run (UTC) | Step | Camera | All 4 sensors on exact frame | Frame time p50 / p95 / max (ms) | Real-time factor | Cleanup | Settings restored | Result |
|---|---|---|---|---|---|---|---|---|
| 02:05:21 | 0.10 s | 800×450 | yes | – | 4.71 | **no** | yes | FAIL |
| 02:08:19 | 0.10 s | 800×450 | yes | – | 4.59 | **no** | yes | FAIL |
| 02:08:40 | 0.10 s | 800×450 | 90/90 | 21.6 / 22.5 / 23.0 | 4.60 | yes | yes | PASS |
| 03:41:42 | 0.05 s | 800×450 | 180/180 | 21.4 / 22.2 / 23.2 | 2.33 | yes | yes | PASS |
| 04:50:31 | 0.10 s | 1280×720 | 90/90 | 26.0 / 26.7 / 27.4 | 3.83 | yes | yes | PASS |

**What this shows (smoke level only):**
- **No missed frames.** Every camera delivered the ticked frame on every
  tick, with no timeouts, no frames ahead of the tick and no stale frames
  dropped, at both step sizes and both resolutions.
- **Tick cost.** Each synchronous tick with the four-camera rig costs about
  22 ms at 800×450 and about 26 ms at 1280×720. This excludes all
  perception and control, so it is not the end-to-end latency. TEST-17
  profiles the full rig.
- **Real-time factor** is simulated time over wall time:
  - 4.6 at 0.10 s, so the simulation runs well ahead of real time;
  - 2.3 at 0.05 s, which leaves half the headroom per simulated second.
- **Depth.** The centre-pixel median was about 40 m, with a 1000 m maximum
  (the sky). Decoded depth is plausible; whether it is optical-axis or ray
  distance is still BL-B-04.
- **Failed runs.** The first two runs failed only the cleanup check (actors
  left behind). Later runs with the same settings passed. Keep an eye on
  cleanup when BL-B-01 builds the real world lifecycle.

## Demo machine

The brief requires the demo to run on a member's own laptop. WCB3LJ64 is a
university desktop, so it is a development and evidence machine, not a demo
candidate.

The primary and backup demo laptops are chosen once Abdullah, Alan and
Zhaolin have recorded their rows. Then:
- repeat the timing on the chosen laptop (TEST-17);
- run the cold-start test there (TEST-21).

| Role | Machine | Owner | Status |
|---|---|---|---|
| Primary demo | – | – | not chosen |
| Backup demo | – | – | not chosen |
