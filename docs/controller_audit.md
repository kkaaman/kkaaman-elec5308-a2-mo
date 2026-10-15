# Forward controller audit and mission design — BL-A-01

Owner: Zhaolin. Reviewer: Abdullah. Status: proposed design; review pending.
Audited source: navigation_control/types.py, controller.py, path.py, steering.py
and speed.py in the supplied A1-derived integrated checkout. No controller
algorithm is rewritten by this task.

## Preserved contracts

| Contract | Existing behaviour | Integration requirement |
|---|---|---|
| Pose | Rear-axle XY metres, yaw radians counterclockwise | Adapter computes rear axle rather than using actor origin |
| Speed | Nonnegative m/s | Reject reverse; do not hide reversing by taking velocity magnitude |
| Steering | Physical equivalent front-wheel angle; positive left | CARLA normalized steer must invert sign and use measured geometry |
| Pedals | [0,1], mutually exclusive | Do not replace physical steering with normalized steer inside core |
| dt | Finite positive seconds | Use fixed simulation delta, not variable wall inference time |
| Route | Copied open polyline; closed paths rejected | Set once per episode; validate legality, continuity and no self crossing before set_path |
| Progress | Monotonic nearest-polyline arc length | Progress alone is not completion or proof of remaining legal route |
| Cross-track diagnostics | Nonnegative Euclidean distance | Retain a separately computed signed CTE for evaluation/HUD |

## Endpoint braking, HOLD and reset

The controller caps speed by `sqrt(2*a*max(0, remaining-margin))`, with a
small endpoint margin defined in controller.py. This caps the requested target;
it is not a physically measured stopping guarantee. Deceleration and lag are
calibrated in BL-E-03. Obstacle braking remains a separate speed-governor policy.

FINISHED requires near-end progress, Euclidean distance to goal within tolerance
and speed at/below stopped threshold. FINISHED is latched and commands zero
throttle with brake. A goal projection reached from far away cannot alone finish.

Away from the goal, target speed zero is a temporary stop. It resets speed PID
history and commands braking; while moving on a curve it preserves lateral
tracking when the target remains ahead. Resuming uses a positive speed request
only after fresh valid perceived clearance. Do not call reset or set_path to HOLD
or resume: doing so erases progress and completion/PID state.

`reset()` retains route/config but clears progress, PID state and finished latch.
`set_path()` validates/copies a new route and then resets. An invalid replacement
leaves the old controller state intact. Repeated set_path in cruise is forbidden
by the mission orchestration policy, not automatically blocked by PathFollower.
Caller exceptions must trigger actuator stopping; the controller does not apply
commands or manage CARLA actors.

## Mission specification

`configs/mission_baseline.json` specifies one fixed forward open mission in
Town10HD_Opt. Tentative sequence: straight approach, legal left junction exit,
straight connector, legal right exit, straight terminal section and stopped goal.
Start and goal must differ; no reversing, route laps or repeated segment crossings.
The map graph exporter (BL-A-02) supplies candidate legal transitions; junction
stitching (BL-A-03) and route building (BL-A-04) choose and validate exact road/lane
IDs and sample coordinates. No arbitrary world points are claimed legal here.
If this manoeuvre sequence is unavailable, document the revised fixed route rather
than forcing an illegal turn or continuously rebuilding from camera markings.

Cruise 5 m/s is provisional from the plan; do not freeze it before vehicle and
stopping-envelope measurements. Actor ego pose/map use still needs tutor scope
confirmation. Obstacle ground truth is evaluation-only.

## Audit risks and checks before live use

ReferencePath rejects exactly closed routes but does not reject all self
intersections or sharp heading jumps. The future route validator must do so.
Global nearest projection can jump to a distant segment on near-self-crossing
routes; the open non-self-crossing mission avoids that ambiguity and retains
progress checks. The inherited CARLA adapter uses unsigned 3D velocity magnitude,
so it cannot prove forward travel. BL-E-01 must measure signed longitudinal speed
and reject unintended reverse rather than passing its magnitude as forward speed.
Physical maximum steering must be separate from the controller cap; vehicle
calibration verifies left/right yaw response on the installed server.

Existing controller tests plus the added multi-cycle mission tests cover sign,
HOLD/resume, finished latch and reset. Numerical success does not claim live
lane following; BL-E-01 and later S0 trials supply that evidence.
