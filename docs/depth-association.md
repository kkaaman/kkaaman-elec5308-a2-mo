# RGB-D association (BL-D-01)

How a detector box becomes a located obstacle. It runs from raw CARLA depth
bytes, through camera and vehicle frames, to the world frame and the route
corridor.

Types and frames are defined in `interfaces.md`. Every numeric threshold
here is an **initial value to validate**. They are frozen at gate G1 in
`configs/acceptance.yaml` (BL-D-06), and changed later only through
`decisions.md`.

## 1. Sensor rig

- The RGB and depth cameras share one transform, resolution and field of
  view. They are spawned as a pair and described by one `CameraModel`.
- Both run every simulator tick in synchronous mode. A `FrameBundle` is
  built only when both images carry the **same** `frame_id`, together with
  the ego pose of that frame. The capture pose is never replaced by a later
  "current" pose.
- If one image is missing at the bounded timeout, the bundle reports that
  sensor invalid, or the tick raises `SensorTimeout`. The stack then stops
  (`interfaces.md`, failure semantics).

## 2. Decoding CARLA depth

CARLA delivers depth as BGRA bytes. Depth is encoded in R, G and B, from
least to most significant, and decoded as:

```text
depth_m = 1000 * (R + 256*G + 65536*B) / 16777215
```

```python
raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(h, w, 4)
b, g, r = (raw[..., k].astype(np.float64) for k in (0, 1, 2))   # cast first
depth_m = (1000.0 * (r + 256.0 * g + 65536.0 * b) / 16777215.0).astype(np.float32)
```

**Cast before multiplying.** `uint8` arithmetic overflows silently.

**Check:** bytes `R=184, G=30, B=5` decode to 20.000 m.

The raw bytes are kept for evidence. Greyscale or log-depth visualisations
are for display only and are never used as metric input.

**Invalid pixels** become `NaN`:
- the sky or far background, at 1000 m or at least the configured
  `max_range_m` (initially 80 m);
- anything closer than `min_range_m` (initially 0.3 m);
- any non-finite value.

## 3. Optical depth or ray distance?

The pinhole maths depends on what CARLA's depth image measures:
- **Optical depth `Z`:** distance along the camera's forward axis.
- **Ray distance `r`:** straight-line distance to the camera centre.

The two differ off-axis by the factor `sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2)`.

**Test (BL-B-04, on the installed CARLA build):**
1. Face the camera squarely at a flat wall, such as a large static prop or
   building face, at a known distance.
2. Read the depth at the image centre and near the left and right edges.

| Result | Convention |
|---|---|
| Edge depth equals centre depth | `optical_z` |
| Edge depth equals centre depth × the factor (for a wall at Z = 10 m, the right edge reads about 14.13 m with a 90° FOV) | `ray` |

Record the result in `CameraModel.depth_kind` and set `depth_kind_verified`.
Until then, code must not assume either convention silently. Use the field.

## 4. From pixel to world: worked example

| Step | Value |
|---|---|
| **Camera** | 800×600, 90° horizontal FOV: `fx = fy = 400`, `cx = 400`, `cy = 300`. Mount `Extrinsics(1.5, 0.0, 2.4)`, no rotation |
| **Pixel** | Image index row 329, column 479. Pixel centres are at `+0.5`, so `u = 479.5`, `v = 329.5` |
| **Depth** | `Z = 20.0 m` (optical) |

1. **Optical frame:**
   ```text
   X = (u − cx)·Z/fx = 3.975 m   (right)
   Y = (v − cy)·Z/fy = 1.475 m   (down)
   Z = 20.0 m
   ```
   If the depth were ray distance, first convert with `Z = r / factor`: here
   `r = 20 m` gives `Z = 19.565 m`. Ignoring the convention would cost
   0.43 m, almost the whole 0.5 m depth-error budget.
2. **Camera body frame** (axis swap from `interfaces.md`):
   ```text
   x = Z = 20.0
   y = −X = −3.975
   z = −Y = −1.475
   ```
3. **Vehicle frame** (apply the mount's rotation, here identity, then its
   translation): `(21.5, −3.975, 0.925)`. That is 21.5 m ahead of the rear
   axle, 3.98 m to the right, and 0.93 m above the ground.
4. **World frame**, using the ego pose **of the same frame**, for example
   `x = 100, y = 50, yaw = 90°`:
   ```text
   x_w = 100 + cos(yaw)·21.5 − sin(yaw)·(−3.975) = 103.975
   y_w =  50 + sin(yaw)·21.5 + cos(yaw)·(−3.975) =  71.5
   ```

**Rotation convention for mounts:**
- Angles are applied yaw, then pitch, then roll, right-hand rule in the
  vehicle frame.
- **Positive pitch tilts the sensor down** (rotation about +y), and positive
  yaw turns it left.
- The CARLA adapter converts CARLA's mount rotation at the boundary. A camera
  pitched down must see the ground nearer; that is the sign check (BL-D-02).

## 5. Box to obstacle

For each detection on a valid frame:

1. **Inner ROI.** Shrink the box to its central 60% in width and 50% in
   height, shifted towards the lower half. This avoids background around the
   silhouette and road surface under it. Pedestrians use the central 40% in
   width.
2. **Sample depth** in the ROI, keeping valid pixels only (§2).
3. **Missing-depth checks:**

   | Check | Initial rule | Outcome |
   |---|---|---|
   | Too few valid pixels | valid fraction < 0.3 | depth invalid → fallback (§6) |
   | Background-dominated | > 50% of valid pixels are more than 3 m beyond the 20th percentile | use the near cluster (pixels within 1.5 m of the 20th percentile), otherwise invalid |
   | Inconsistent | IQR of the used pixels > max(1.0 m, 0.15 × median) | keep, with `range_sigma_m` = IQR / 1.35 |

4. **Range.** The median of the used pixels is the distance to the visible
   surface. `range_sigma_m` = max(0.1 m, IQR / 1.35), plus the convention
   error if `depth_kind_verified` is false.
5. **Footprint in the vehicle frame:**
   - The lateral edges come from the box columns at the measured depth:
     `X_left = (x_min − cx)·Z/fx` and `X_right = (x_max − cx)·Z/fx`, which
     become vehicle `y` through `y = −X`.
   - The **near face** is placed at the measured surface.
   - The footprint extends **away** from the sensor by a class prior,
     because only the front surface is visible: vehicle length 4.5 m,
     pedestrian 0.6 m, cyclist 1.8 m, unknown 1.0 m.
6. **World footprint** uses the capture-frame ego pose (§4, step 4).
7. **Identity.** Use the namespaced `track_key`. Untracked detections pass
   through, but cannot be confirmed across frames.

**Example:** a car straight ahead at 20 m spans image columns 382 to 418:
- Lateral extent is ±0.9 m, so it is 1.8 m wide.
- In the vehicle frame its near face is at `x = 21.5`, and the footprint
  covers `x ∈ [21.5, 26.0]`, `y ∈ [−0.9, 0.9]`.

## 6. Fallbacks and missing depth

Unknown distance **never** turns a detection into "no obstacle".

| Situation | What the source reports |
|---|---|
| Depth invalid, box bottom below the horizon | `RangeMethod.GROUND_PROJECTION` from the box's bottom edge (flat-ground model), `range_sigma_m` ≥ 15% of range |
| Depth invalid, known-width class | `RangeMethod.KNOWN_WIDTH`: car 1.8 m, person 0.5 m. `range_sigma_m` ≥ 25% of range |
| No estimate possible | Obstacle omitted, and the source reports `SourceHealth.DEGRADED` with the reason. Behaviour treats this as "cannot certify clear" (BL-D-05) |
| Depth image missing for the frame | `SourceHealth.FAILED`. The stack stops |

## 7. Route corridor and gap

- **Corridor.** The swept area of the ego along the mission route: lateral
  half-width = ego half-width + 0.3 m margin. It runs from the front bumper
  forward to the look-ahead distance, initially at least 40 m, and always
  beyond the stopping budget below.
- **Projection.** Each world-footprint corner is projected onto the route
  polyline. The projection is restricted to within ±15 m of current
  progress, so it cannot jump to a later part of the route. That gives
  `route_s_m = (s_min, s_max)` and `route_d_m = (d_min, d_max)`.
- **In corridor.** The `route_d_m` interval, inflated by 2 × `range_sigma_m`,
  overlaps the corridor half-width, **and** `s_max` lies ahead of the front
  bumper.
- **Gap.**
  ```text
  gap_m = s_min − (s_ego + rear_axle_to_front_bumper_m) − 2·range_sigma_m
  ```
  It is measured along the route from the ego's front bumper to the nearest
  obstacle surface, conservatively.
- **Stopping budget** (BL-D-06 freezes the inputs after calibration):
  ```text
  required_gap = v·τ + v²/(2·a_dec) + d_stop + uncertainty
  ```
  Example, illustrative only: 5 m/s, τ = 0.2 s, 1.2 m/s², d_stop = 3.5 m →
  14.9 m before the uncertainty term.

## 8. What is tested where

| Spec | Test |
|---|---|
| Depth decode, overflow | unit (BL-D-03) |
| Optical vs ray convention | CARLA (BL-B-04), then a unit fixture |
| Pixel → vehicle → world, sign checks at several headings | unit (BL-D-02) |
| ROI, background, inconsistency, fallbacks | unit with synthetic depth (BL-D-03) |
| Corridor membership and gap | unit (BL-D-04), harness (BL-F-06) |
| Depth error ≤ 0.5 m within 25 m | CARLA dataset (BL-B-04), see `acceptance.md` |
