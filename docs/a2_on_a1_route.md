# A2 on A1's route

A2 drives the roads that A1 drove in its TEST-01 run, so the two can be compared on
the same roads: A1 from its recording, A2 live. This is extension work on top of the
frozen baseline. The baseline mission, its frozen configuration and its 30 acceptance
runs are unchanged.

## The route

`configs/missions/a1_test01.json` defines it. The lanes were matched to A1's logged
poses (`docs/evidence/TEST-01-samples.csv`) with a nearest-lane search that only accepts
lanes running in A1's direction of travel.

| | A1 (TEST-01 log) | A2 route |
|---|---|---|
| Start | (102.24, -91.78), road 1 lane 1 | same point, on the lane centre |
| Roads | 1 → 8 → 4 → junction 468 (straight) → 5 | 1:0:1 → 8:0:1 → 4:0:-1 → 515:0:-1 → 5:0:-1 |
| Lanes | changed lanes 1↔2 on roads 1 and 8, and -2↔-1 on road 5 | stays in lane 1, then lane -1 (A1's start and end lanes); A2 has no lane-change manoeuvre |
| End | (-98.21, -117.95), after 120 s | A1's end point projected onto lane -1 |
| Length | 226 m driven | 228.9 m |

Road 1 begins with a 62 m bend (radius about 35-48 m) that turns the car through about
90°. It is followed by a 125 m straight, junction 468 (straight
through) and a gentle curve on road 5. A1's own log reports the nearest lane id without
checking direction. A direction-aware match places A1's final position in lane -1 of
road 5, which runs the same way, not in the opposing lane.

The chain is checked like the automatic mission:
- every transition must be a listed successor;
- each junction crossing must be the unique legal exit of its declared manoeuvre;
- the route must not cross itself, and start and goal must be at least 30 m apart.

`prepare_mission.py --route` then validates the route live, as LiveSetup does: every
sample must be on a mission lane within 0.20 m, and lanes must be at least 2.5 m wide.

## Obstacle placement

On this route, S1-S4 place their obstacle at s = 120 m (`obstacle_s_m` in the route
definition), mid-way along the straight. S5 still places its obstacle 6 m after the
junction exit (s = 205 m). The scenario YAMLs and the baseline mission are unchanged.

The scenarios' own s = 40 m would land in the middle of the opening bend, where the road
runs at about 45° to the map axes. There the track filter's world-axis footprint box
widens a car beside the lane into the corridor:
- with a roadside car (S3), the harness held A2 permanently;
- with a car in the lane (S1), A2 stopped 6.17 m short, against the 6.0 m limit.

On the straight all six scenarios pass in the harness, with stop margins of 4.5 m
(S1, S2, S4) and 5.25 m (S5). `tests/integration/test_a1_route.py` keeps the bend case
as a strict expected failure. It will start passing once footprints follow the road
rather than the map axes. The same limitation applies to parked map cars on any
diagonal section of a route, so watch the first live S0 run through the bend and after
the junction.

## Running it

Use a separate state file, so the baseline state keeps its mission. The new state
starts as a copy of the baseline state:
- depth calibration, braking, event verification and the tuned candidate carry over,
  because they belong to the map and the vehicle;
- the frozen configuration and everything driven with it are dropped, because they
  belong to the baseline mission.

Runs on this route therefore use the tuned candidate configuration (same runtime
parameters, status PROVISIONAL).

```powershell
$state = "$HOME\Downloads\Group7_a1route_state.json"
.\scripts\run_batch35.ps1 -Stage RouteSetup -Route configs\missions\a1_test01.json -StateFile $state
.\scripts\run_batch35.ps1 -Stage Quick -StateFile $state
.\scripts\run_batch35.ps1 -Stage Navigation -StateFile $state
```

- RouteSetup validates the route live and opens `mission.png`.
- Quick runs S0 navigation-only, S0 full stack and S1 on the route, plus the
  lane-monitor replay.
- Navigation runs five S0 navigation-only runs, the input for TEST-03 on the same roads.

Watch one run in CARLA:

```powershell
$s = Get-Content $state -Raw | ConvertFrom-Json
.venv\Scripts\python.exe scripts\run_scenario.py --engine carla --scenario S2 --config $s.candidate --mission $s.mission --depth-calibration $s.depth --event-verification $s.event_verification --follow --output "$HOME\Downloads\Group7_a1route_S2_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

## Results (21 Oct)

The route validated live, and the obstacle spawned once the ground search started just
above the road (`fix(carla)` commit).

**Quick:** S0 navigation-only, S0 full stack and S1 all collected.

**Lane monitor (RV-06) on this route:**
- agreed with the map's curve direction in 97.5% of usable frames (489/517 and 580/605);
- median offset 6 cm;
- for comparison, 22-26% on the baseline route's unmarked junction turn.

**Navigation:** five runs, `MEASURED_PASS_REVIEW_REQUIRED`.

**TEST-03 on the same roads:** A2 mean |CTE| 0.025 m (max 0.085 m) against A1's 0.918 m
(max 3.43 m). Details, the distance-aligned figure and the remaining differences are in
`docs/evidence/TEST-03-a1-route/`.

## What the comparison shows, and what it does not

With A2 on the same roads, the route difference in TEST-03 goes away. These differences
remain:
- **Pose source.** A2 uses the simulator's vehicle pose; A1 used camera lane
  estimates.
- **Timing.** A2 is synchronous at 0.05 s; A1 was asynchronous, at about 0.43 s.
- **Speed.** A2 cruises at 4.8 m/s; A1 at 3 m/s.
- **Vehicle geometry.**
- **Lane changes.** A1 changed lanes where A2 holds one lane.

Traffic lights are frozen green in A2 runs, and A1 never stopped for one in TEST-01.
