# Forward parking P1 — E5-01 to E5-03

P1 is an opt-in, simulator-independent forward parking episode. Given an explicit
rear-axle goal, rectangular bay, convex manoeuvring area and ego dimensions, it
plans and follows a bounded approach, stops for blocked or unobserved swept
volumes, and changes DRIVE to PARK once all endpoint constraints hold.

This implementation targets issue #59's synthetic/harness acceptance. The
baseline CARLA route and G1 admission remain separate. Live CARLA parking is
**NOT RUN**. The harness depth rig and body dimensions are illustrative; passing
P1 does not certify the existing front camera's near-field coverage.

## Run and inspect

Use the project Python >=3.10 environment (the Windows project venv uses 3.12):

```powershell
.\.venv\Scripts\python.exe scripts/run_p1_acceptance.py --output runs/P1 --plots
.\.venv\Scripts\python.exe -m pytest tests/unit/test_parking.py tests/integration/test_parking.py
```

`--plots` additionally needs matplotlib; omit it for the minimal CI environment.
Each invocation creates a new timestamped batch. Every case has `samples.csv`,
`summary.json` and, if execution reaches the end, `plan.json`. Optional
`trajectory.png` shows the planned/driven rear axle, body rectangles, bay and
speed history. Samples are streamed so exceptions retain the partial trace.
The batch retains all case summaries, including failures, and a manifest with
configuration, exact source-byte hashes, Git revision and dirty status. The
process exits nonzero if any selected case fails. CI retains the complete batch.

The checked configuration is `configs/parking_p1.json`. Its independent parking
policy does not load or enable the G1-gated baseline operating profile. The
curated [P1 evidence record](evidence/E5-P1/README.md) reports the measured scope.

## Interface contract

Public entry points:

| Type | Meaning |
|---|---|
| `autonomy.parking_contracts.MotionState` | Rear-axle x/y/yaw and **signed longitudinal** speed; default mode DRIVE |
| `ParkingBay` | Oriented rectangular bay with independent centre, heading, length and width |
| `ParkingGoal` | Rear-axle target and final heading, bay, clearance and endpoint tolerances |
| `ParkingVehicle` | Wheelbase, controller steering cap, rear-axle-to-bumper distances, half width and height |
| `ParkingConfig` | Explicit parking speed, tracking, sensor and braking policy |
| `autonomy.parking.ParkingEpisode` | One installed path and one DRIVE → PARK transition per episode |
| `ParkingStep` | Expiring `ControlPacket`, motion/mode, phase and metric diagnostics |

World coordinates are the existing right-handed frame (+Y left, +Z up), angles
are counterclockwise radians, and lengths/speeds are metres and m/s. A1
`VehicleState` still rejects negative speed; `ControlCommand` has no changed
fields. `MotionState.as_forward()` explicitly rejects reverse. Adding the new
signed representation does not enable reverse execution or change A1 imports.

PARK is an application mode with a latched full-brake command. It is not a CARLA
gear/handbrake call. Construct a new episode for a new mission; no automatic
release from PARK or repeated path installation is performed.

Example integration, after obtaining a valid initial `FrameBundle`:

```python
from autonomy.parking import ParkingEpisode
from autonomy.parking_contracts import ParkingBay, ParkingGoal, ParkingVehicle

vehicle = ParkingVehicle(3.00465, 0.3490658503988659, 4.0, 1.0, 0.9, 1.7)
bay = ParkingBay(17.5, 0.0, 0.0, 7.0, 3.8)
goal = ParkingGoal(16.0, 0.0, 0.0, bay)
area = ((-5.0, -5.0), (25.0, -5.0), (25.0, 5.0), (-5.0, 5.0))
parking = ParkingEpisode(initial_frame, goal, vehicle, area)

# In the capture owner's fixed-step loop:
result = parking.step(frame)
next_frame = world.step(result.packet)
```

These numerical dimensions are fixture values. A real adapter must provide its
validated ego body dimensions and calibrated observation rig. Core code imports
neither `carla` nor `carla_adapter`. The initial frame must be the first frame
stepped; later frames must belong to the same episode and advance by one fixed
timestep. The world/adapter must enforce both command expiry clocks. Replay
can explicitly supply `now_monotonic_s`; production uses the host clock and
also checks processing time against the command TTL. Malformed non-FrameBundle
inputs raise; callers must apply their usual braking-on-exception boundary.
Observation age is checked again at the end of computation. A late result
latches a fault and cannot publish a newly calculated PARK transition.

## Path and full-body bounds

The planner tries a deterministic finite set of quintic Bezier approaches with
matching endpoint tangents and zero endpoint curvature, followed by a straight
alignment tail. A straight approach is handled directly. Derivative controls
must all advance along the approach chord, excluding loops and self crossings.
Some geometrically reachable bays fall outside this bounded family and are
rejected with no motion; P1 is not a general-purpose parking-space search.

The curvature limit is `tan(controller_steering_limit) / wheelbase`. At the
fixture's 20-degree cap and 3.00465 m wheelbase the minimum radius is about
8.26 m. The 70-degree physical steering specification is not used to enlarge
the controller's feasible set.

De Casteljau subdivision bounds curvature over each **whole** interval using
the convex hulls of first and second derivatives and a positive lower bound on
tangent speed. Acceptance does not rely solely on point samples. Each interval's
body sweep encloses its entire position control hull and all corner rotations,
then adds goal clearance and a bounded tracking envelope. Sweeps must fit the
supplied convex area; the final full rectangle plus clearance must fit the bay.
Runtime deviations beyond the configured position/heading tracking envelope
latch a fault stop. The harness separately checks continuous per-tick body
envelopes for contact and area exit; `KinematicWorld` has no collision response.

## Depth-derived clearance

`DepthParkingClearance` projects each swept body prism into the aligned depth
buffer. All pixels in the outward-rounded projected bounding rectangle must
have valid, in-range depth beyond the prism's farthest axial extent plus the
configured error allowance. This is intentionally conservative: obstacles,
occlusion and ground intersections can stop the car earlier than necessary.
Off-image volumes, NaNs and absent depth produce UNKNOWN, never clear space.
An empty detection batch is not used as clearance evidence. The core does not
consume evaluation actor boxes. LiDAR is not implemented in this change.

The supported sensor model assumes a known local flat ground, bounded depth
error and sufficient spatial resolution. With the default ground exclusion,
the checked prism starts 0.08 m above that plane and extends to ego body height;
smaller hazards and overhangs outside that height are outside this profile.
Depth margin is 0.02 m in the ideal fixture, not a live depth calibration.
No accumulated map, ego-occlusion mask or unseen-space inference is used.
Consequently a typical forward-only camera that cannot see the rear/side sweep
will hold. Integrating an actual near-field rig requires coverage and error
qualification, possibly additional sensors; it cannot be done by declaring
unknown pixels free.

The harness supplies an ideal 256×256 downward depth camera, 8 m above the local
ground, 120-degree FOV, with conservative projected-box rasterization and no
ego rendering. Its scene boxes are confined to the renderer and evaluator.
P1 sees only the resulting `FrameBundle`. All-NaN dropout fixtures are distinct
from observed ground. This demonstrates the data/control boundary and algorithm;
it does not validate real camera mounting, occlusion, sensor noise or physics.

## Stops, completion and handover

The default maximum requested speed is 0.8 m/s for the whole parking approach;
configuration cannot exceed 1 m/s. The existing speed governor combines braking
distance, observation/delivery response budget and parking-specific margins.
Endpoint slowing uses the existing path follower with a shorter lookahead.
Normal braking keeps path following; emergency or unknown coverage issues full
brake. A stopped blocker yields HOLD, not PARK.

After a blocker, `ClearanceGuard` requires at least five consecutive valid clear
frames **and** at least 0.4 seconds. Invalid data resets partial confirmation and
preserves the blocked latch. Temporary holds do not reinstall the path or reset
episode progress. Frame identity/time violations, excessive tracking error,
timeout and controller failures latch FAULT with a retained reason.

The path follower receives completion permission only when all of these hold:

- rear-axle position error <=0.3 m (or a stricter configured limit);
- shortest-angle heading error <=5 degrees (or a stricter configured limit);
- full body plus clearance lies inside the bay;
- current swept-volume evidence is valid and clear, with no pending clearance hold.

The follower additionally requires sufficient route progress and measured speed
<=0.05 m/s by default. Only then does the episode record `drive_to_park` once and
hold full brake. Position alone, progress=100%, missing tracks, invalid depth,
or an incorrect heading cannot complete P1. A stopped invalid endpoint faults
instead of claiming success. Later PARK commands do not re-evaluate clearance
and are labelled `not_rechecked`; they continue braking.

## Verification and traceability

E5-01 is covered by the opt-in contracts, strict tolerance validation, A1 negative
speed regression and default DRIVE test. E5-02 is covered by analytic interval
bounds, intermediate full-body checks, rotated/left/right/straight closed loops,
and rejection of narrow or curvature-infeasible bays. E5-03 covers blocked and
side-blocked bays, hold/resume, stale/missing/out-of-view depth, frame faults,
near-goal heading/speed/body guards and the once-only PARK latch.

The Windows baseline hash issue is fixed by explicit LF attributes for the 30
existing files byte-pinned by the BL-D-06 evidence review. Their contents match
Git's LF blobs. No checksum, evidence threshold or runtime baseline setting is
changed to obtain a pass.
