# Module notes and source surface - DOC-05

Generated from Python source without importing models or simulator libraries.
Purpose text comes from module docstrings or curated source notes. An inventory is not a review of
every design rationale; missing purpose notes require owner follow-up.

## Public protocol and data flow

The adapter converts simulator state and sensor data into plain contracts.
Perception supplies frame-associated observations; autonomy checks freshness
and operating limits; planning supplies control targets; the adapter applies
commands. The offline harness provides a separate world implementation.
See [interfaces](interfaces.md), [detector contracts](detector_contracts.md),
[simulator boundary](simulator_boundary.md) and [API](api.rst).

## `src/autodrive/perception/lane_dataset.py`

Loads paired lane images and masks for training; coordinated resizing and tensor conversion keep labels aligned with input pixels.

Public definitions: `LaneDataset`.

Direct module imports: `cv2`, `pathlib`, `torch`, `torch.utils.data`.

## `src/autodrive/perception/lane_inference.py`

Legacy BGR interface. New callers should use perception.lane_inference.

Public definitions: `LaneSegmenter`.

Direct module imports: `cv2`, `numpy`, `perception.lane_inference`.

## `src/autodrive/perception/lane_path.py`

Legacy import compatibility. Implementation lives in perception.lane_path.

Public definitions: none at module level.

Direct module imports: `perception.lane_path`.

## `src/autodrive/perception/lane_segmentation.py`

Legacy model imports. Implementation lives in perception.lane_segmentation.

Public definitions: none at module level.

Direct module imports: `perception.lane_segmentation`.

## `src/autodrive/perception/losses.py`

Defines focal and soft Dice training losses to address difficult pixels and mask overlap alongside classification loss.

Public definitions: `FocalLoss`, `SoftDiceLoss`, `CombinedLoss`.

Direct module imports: `torch`, `torch.nn`, `torch.nn.functional`.

## `src/autonomy/__init__.py`

Contracts, the autonomy stack and safety arbitration (workstream F).

Public definitions: none at module level.

Direct module imports: none.

## `src/autonomy/admission.py`

Verify the versioned G1 configuration and source identity before loading.

Public definitions: `read_json`, `config_identity`, `source_inventory`, `scope_fingerprint`, `load_admitted_baseline`.

Direct module imports: `hashlib`, `json`, `pathlib`.

## `src/autonomy/baseline_operation.py`

Load the frozen, bounded operating profile without bypassing its gate.

Public definitions: `frozen_governor_config`, `baseline_operational_stack`.

Direct module imports: `autonomy.admission`, `autonomy.operational`, `autonomy.stack`, `math`, `pathlib`, `planning.speed_governor`.

## `src/autonomy/detector_contracts.py`

BL-C-01 protocols using the BL-F-01 frame contract.

Public definitions: `Detector`, `ObstacleSource`.

Direct module imports: `interfaces`, `obstacle_perception.types`, `typing`.

## `src/autonomy/freshness.py`

Operational freshness gate using existing stack and command contracts.

Public definitions: `OperationalFreshnessGate`.

Direct module imports: `math`.

## `src/autonomy/interfaces.py`

Version 1 simulator-independent packets (BL-F-01).

Public definitions: `EgoPose`, `CameraCalibration`, `FrameBundle`, `GraphNode`, `GraphEdge`, `LaneGraph`, `Diagnostics`, `ControlPacket`, `World`, `AutonomyStack`.

Direct module imports: `__future__`, `dataclasses`, `math`, `navigation_control.types`, `numpy`, `typing`.

## `src/autonomy/mission.py`

One fixed route installation per episode, with explicit temporary holds.

Public definitions: `MissionEvent`, `MissionEpisode`.

Direct module imports: `dataclasses`, `math`, `navigation_control`, `numbers`, `re`, `typing`.

## `src/autonomy/obstacle_policy.py`

Freshness-aware composition of existing tracking, clearance and governance.

Public definitions: `CorridorObservation`, `TrackedGovernancePolicy`.

Direct module imports: `dataclasses`, `navigation_control.types`, `planning.behaviour`.

## `src/autonomy/operational.py`

Explicit startup entry point for freshness-gated tracked governance.

Public definitions: `operational_stack`.

Direct module imports: `autonomy.freshness`, `autonomy.stack`, `perception.frame_detector`.

## `src/autonomy/paths.py`

Repository locations that do not depend on the current working directory.

Public definitions: `config_path`, `model_path`, `run_dir`.

Direct module imports: `__future__`, `os`, `pathlib`.

## `src/autonomy/stack.py`

Validated, simulator-free stack skeleton with latched emergency stopping.

Public definitions: `StackConfig`, `SkeletonStack`.

Direct module imports: `dataclasses`, `interfaces`, `json`, `math`, `navigation_control.types`, `obstacle_perception.types`, `pathlib`, `time`.

## `src/carla_adapter/__init__.py`

CARLA integration adapter: the only package allowed to import ``carla``.

Public definitions: none at module level.

Direct module imports: none.

## `src/carla_adapter/a1_capture.py`

Observational TEST-01 recorder around the inherited A1 demo.

Public definitions: `main`.

Direct module imports: `argparse`, `csv`, `dataclasses`, `evaluation.a1_capture`, `hashlib`, `importlib.util`, `inspect`, `json`, `math`, `pathlib`, `platform`, `subprocess`, `sys`, `time`, `traceback`, `zipfile`.

## `src/carla_adapter/adapter.py`

CARLA-specific translation between CARLA and navigation_control types.

Public definitions: `carla_vehicle_to_state`, `command_to_carla_control`, `apply_command`.

Direct module imports: `carla`, `math`, `navigation_control.types`.

## `src/carla_adapter/conversions.py`

Raw CARLA BGRA camera decoding without importing the simulator.

Public definitions: `bgra_array`, `decode_rgb`, `decode_depth`, `axial_depth`.

Direct module imports: `numpy`.

## `src/carla_adapter/dataset_capture.py`

Stationary, teleported training views; never an autonomy acceptance run.

Public definitions: `FrameBuffer`, `transform_dict`, `target_surface_reference`, `location_for_target`, `main`.

Direct module imports: `PIL`, `argparse`, `collections`, `datetime`, `evaluation.dataset`, `hashlib`, `json`, `math`, `numpy`, `pathlib`, `random`, `threading`, `time`.

## `src/carla_adapter/depth_calibration.py`

Measure raw depth convention against off-axis geometry ray intersections.

Public definitions: `ray_reference`, `choose_convention`, `collect`, `main`.

Direct module imports: `argparse`, `carla_adapter.conversions`, `carla_adapter.world`, `hashlib`, `json`, `math`, `numpy`, `pathlib`, `queue`, `time`.

## `src/carla_adapter/longitudinal.py`

BL-E-03 throttle/brake step collection, bounded to a checked straight site.

Public definitions: `collect`, `main`.

Direct module imports: `argparse`, `carla_adapter.vehicle`, `carla_adapter.world`, `csv`, `evaluation.longitudinal`, `json`, `math`, `pathlib`.

## `src/carla_adapter/s0_runner.py`

Run one frozen, obstacle-free S0 mission and retain auditable evidence.

Public definitions: `configuration_digest`, `execution_provenance`, `signed_cross_track`, `route_speed_target`, `CameraReceipt`, `RGBReceiptQueue`, `RunSensors`, `FrozenEnvironment`, `run_s0_episode`, `run_s0_suite`.

Direct module imports: `autonomy.mission`, `collections`, `copy`, `csv`, `dataclasses`, `evaluation.s0_metrics`, `hashlib`, `importlib.metadata`, `json`, `math`, `navigation_control`, `navigation_control.path`, `pathlib`, `platform`, `subprocess`, `sys`, `threading`, `time`, `traceback`, `vehicle`, `world`.

## `src/carla_adapter/scenario_verification.py`

Live BL-B-06/07 checks: scenario placement, evaluation truth and event sensors.

Public definitions: `gap_cases`, `Verification`, `source_provenance`, `collect`, `main`.

Direct module imports: `argparse`, `dataclasses`, `evaluation.scenario_events`, `evaluation.scenarios`, `hashlib`, `json`, `math`, `navigation_control`, `navigation_control.path`, `navigation_control.types`, `pathlib`, `planning.baseline_mission`, `scenario_world`, `subprocess`, `sys`, `vehicle`, `world`.

## `src/carla_adapter/scenario_world.py`

Live S1-S5 obstacles, evaluation truth and event sensors (BL-B-06/07).

Public definitions: `core_xy`, `route_z`, `footprint_core`, `ScenarioObstacle`, `ego_front_offset`, `ego_truth`, `ScenarioEventSensors`, `event_records`.

Direct module imports: `evaluation.scenario_events`, `evaluation.scenarios`, `evaluation.stop_truth`, `math`, `threading`.

## `src/carla_adapter/sensors.py`

Bounded exact-frame RGB-D join; callbacks never tick the simulator.

Public definitions: `CameraRig`, `AlignedCapture`, `FrameJoiner`, `RGBDCapture`.

Direct module imports: `autonomy.interfaces`, `carla_adapter.conversions`, `collections`, `dataclasses`, `math`, `numpy`, `threading`, `time`.

## `src/carla_adapter/steering_calibration.py`

Short fixed-step Tesla steering sweeps through the installed vehicle adapter.

Public definitions: `main`.

Direct module imports: `argparse`, `csv`, `dataclasses`, `evaluation.steering_calibration`, `hashlib`, `json`, `math`, `navigation_control.types`, `pathlib`, `subprocess`, `sys`, `time`, `vehicle`, `world`.

## `src/carla_adapter/topology.py`

CARLA 0.9.16 topology queries and coordinate conversion (BL-A-02).

Public definitions: `TopologyError`, `export_lane_topology`, `validate_lane_topology_on_map`, `connect_map`, `load_opendrive`.

Direct module imports: `__future__`, `hashlib`, `math`, `planning.lane_graph`, `typing`, `xml.etree.ElementTree`.

## `src/carla_adapter/vehicle.py`

Fixed Tesla adapter. CARLA imports stay inside this package.

Public definitions: `FixedVehicleAdapter`, `main`.

Direct module imports: `math`, `time`, `vehicle_geometry`.

## `src/carla_adapter/vehicle_geometry.py`

Validated fixed-vehicle geometry and simulator-free coordinate conversions.

Public definitions: `VehicleGeometry`, `geometry_from_wheels`, `forward_state`, `normalized_steering`.

Direct module imports: `dataclasses`, `math`, `navigation_control.types`, `numpy`.

## `src/carla_adapter/world.py`

Sensor-free synchronous world ownership and cleanup for BL-B-01.

Public definitions: `WorldTiming`, `SynchronousWorldSession`, `main`.

Direct module imports: `dataclasses`, `math`, `threading`.

## `src/cloud/__init__.py`

Optional hybrid inference backend (extension E4). Off by default.

Public definitions: none at module level.

Direct module imports: none.

## `src/demo_combined_perception.py`

Combined live perception overlay: segmentation + object detection +

Public definitions: `make_cached_backend`, `draw_object_detections`, `draw_traffic_detections`, `draw_layer_hud`, `build_combined_overlay`, `measure_straightness`, `rank_straight_spawn_candidates`, `main`.

Direct module imports: `argparse`, `carla`, `collections`, `cv2`, `numpy`, `overlay_enhancements`, `perception`, `perception.lane_inference`, `perception.lane_path`, `segmentation.segmentation`, `sys`, `time`, `traffic_perception`, `typing`.

## `src/demo_fused_lane_following.py`

Fused demo: lane detection drives the vehicle during ordinary driving,

Public definitions: `measure_straightness`, `rank_straight_spawn_candidates`, `advance_past_junctions`, `make_cached_backend`, `draw_object_detections`, `draw_traffic_detections`, `draw_layer_hud`, `main`.

Direct module imports: `argparse`, `carla`, `carla_adapter.adapter`, `cv2`, `ground_projection`, `lane_to_waypoints`, `math`, `navigation_control.controller`, `navigation_control.types`, `numpy`, `overlay_enhancements`, `perception`, `perception.lane_inference`, `perception.lane_path`, `segmentation.segmentation`, `sys`, `time`, `traffic_light_color`, `traffic_perception`, `turn_decision`, `typing`.

## `src/evaluation/__init__.py`

Run logging, metrics and evaluation-only ground truth (workstreams B and F).

Public definitions: none at module level.

Direct module imports: none.

## `src/evaluation/a1_capture.py`

Simulator-free TEST-01 raw logging and explicitly defined summary metrics.

Public definitions: `normalized_hash`, `verify_sources`, `signed_lane_cte`, `metrics`, `CaptureLog`, `write_json`.

Direct module imports: `csv`, `hashlib`, `json`, `math`, `pathlib`.

## `src/evaluation/a1_comparison.py`

TEST-03: compare A1 and A2 navigation on one cross-track definition.

Public definitions: `LaneCentrelines`, `load_a1_capture`, `a1_metrics`, `a2_metrics`, `compare`.

Direct module imports: `csv`, `json`, `math`, `numpy`, `pathlib`.

## `src/evaluation/dataset.py`

Dataset contracts and metrics; no simulator or model imports.

Public definitions: `sha256`, `save_json`, `local_file`, `road_splits`, `decode_bgra`, `decode_depth`, `visible_boxes`, `confusion`, `metrics`, `read_dataset`.

Direct module imports: `PIL`, `hashlib`, `json`, `math`, `numpy`, `pathlib`.

## `src/evaluation/longitudinal.py`

Measured braking budgets; absent/failed trials cannot justify a speed cap.

Public definitions: `braking_summary`.

Direct module imports: `math`.

## `src/evaluation/masking.py`

BL-B-05 decision rule for optional semantic masking of box depth.

Public definitions: `masking_decision`.

Direct module imports: `math`, `numpy`.

## `src/evaluation/metrics.py`

Offline S0 metrics. Missing evidence is blocked, never interpreted as zero.

Public definitions: `percentile`, `first_sustained`, `stopped_margin`, `read_json`, `validate_s0_criteria`, `evaluate_run`, `aggregate`.

Direct module imports: `__future__`, `json`, `math`, `pathlib`.

## `src/evaluation/run_log.py`

Versioned JSONL run logs; host times use one relative monotonic origin.

Public definitions: `plain`, `write_json`, `provenance`, `RunLogger`.

Direct module imports: `__future__`, `collections.abc`, `dataclasses`, `enum`, `hashlib`, `importlib.metadata`, `json`, `pathlib`, `platform`, `subprocess`.

## `src/evaluation/s0_metrics.py`

Frozen S0 acceptance metrics with explicit evidence and failure retention.

Public definitions: `evaluate_s0_run`, `aggregate_s0_runs`.

Direct module imports: `collections.abc`, `math`, `statistics`.

## `src/evaluation/scenario_events.py`

Collision and lane-invasion event records for S0-S5 (BL-B-07). Simulator-free.

Public definitions: `CollisionContacts`, `collision_record`, `lane_invasion_record`, `validate_exclusions`, `legal_exclusion`, `count_contacts`, `count_unplanned_invasions`.

Direct module imports: `math`.

## `src/evaluation/scenarios.py`

Deterministic S1-S5 obstacle scenarios (BL-B-06). Simulator-free and evaluation-only.

Public definitions: `Obstacle`, `Removal`, `ScenarioDefinition`, `load_scenario`, `load_scenarios`, `route_heading`, `placement_pose`, `RemovalSchedule`.

Direct module imports: `dataclasses`, `hashlib`, `json`, `math`, `pathlib`.

## `src/evaluation/steering_calibration.py`

SI steering-response analysis; no CARLA imports or automatic adapter tuning.

Public definitions: `inferred_angle`, `SpeedServo`, `analyze`.

Direct module imports: `csv`, `dataclasses`, `json`, `math`, `pathlib`, `statistics`.

## `src/evaluation/stop_truth.py`

Evaluation-only stop-margin truth and HOLD/resume metrics (BL-B-07). Simulator-free.

Public definitions: `ego_front_offset_m`, `footprint_samples`, `signed_lateral_m`, `route_surface_gap`, `hold_metrics`.

Direct module imports: `math`.

## `src/geometry/__init__.py`

Camera geometry, rigid transforms and ground projection.

Public definitions: none at module level.

Direct module imports: none.

## `src/geometry/camera.py`

Pinhole optical rays, axial-depth projection and local ground intersection.

Public definitions: `CameraIntrinsics`, `pixel_ray`, `pixel_depth_to_ego`, `ego_point_to_pixel`, `pixel_to_ground`.

Direct module imports: `autonomy.interfaces`, `dataclasses`, `math`, `numpy`, `transforms`.

## `src/geometry/ground_projection.py`

A1-compatible horizontal-camera helpers, backed by canonical geometry.

Public definitions: `pixel_to_vehicle_local`, `project_ground_point_to_pixel`.

Direct module imports: `camera`, `math`, `transforms`.

## `src/geometry/transforms.py`

Right-handed SI rigid transforms. Simulator axes are converted at its boundary.

Public definitions: `rigid_matrix`, `transform_point`, `inverse_rigid`, `ego_to_world_matrix`, `vehicle_local_to_world`.

Direct module imports: `math`, `numpy`.

## `src/ground_projection.py`

Compatibility exports for the canonical geometry.ground_projection helpers.

Public definitions: none at module level.

Direct module imports: `geometry.ground_projection`.

## `src/lane_to_waypoints.py`

Convert a lane_path.extract_ego_path() result into a short, real-world

Public definitions: `build_local_route`.

Direct module imports: `ground_projection`, `math`.

## `src/navigation_control/__init__.py`

Forward-only path following in metres, seconds and radians.

Public definitions: none at module level.

Direct module imports: `controller`, `types`.

## `src/navigation_control/_validation.py`

Small numeric validation helpers shared by the control components.

Public definitions: `finite`, `nonnegative`, `positive`, `clamp`.

Direct module imports: `math`, `numbers`.

## `src/navigation_control/controller.py`

Coordinate path progress, lateral control, longitudinal control and stopping.

Public definitions: `PathFollower`.

Direct module imports: `_validation`, `math`, `path`, `speed`, `steering`, `types`, `typing`.

## `src/navigation_control/path.py`

Polyline validation, projection and arc-length interpolation.

Public definitions: `Projection`, `ReferencePath`.

Direct module imports: `_validation`, `bisect`, `dataclasses`, `math`, `types`, `typing`.

## `src/navigation_control/speed.py`

Discrete speed PID with conditional-integration anti-windup.

Public definitions: `SpeedPID`.

Direct module imports: `_validation`, `types`, `typing`.

## `src/navigation_control/steering.py`

Geometric Pure Pursuit lateral controller.

Public definitions: `PurePursuit`.

Direct module imports: `_validation`, `math`, `types`.

## `src/navigation_control/types.py`

Immutable public input, output and configuration types.

Public definitions: `VehicleState`, `Waypoint`, `ControllerConfig`, `ControlStatus`, `ControlCommand`, `ControlDiagnostics`, `ControlResult`.

Direct module imports: `_validation`, `dataclasses`, `enum`, `math`.

## `src/obstacle_perception/__init__.py`

Obstacles from detections, depth and LiDAR (workstreams B, C and D).

Public definitions: none at module level.

Direct module imports: none.

## `src/obstacle_perception/box_depth.py`

Robust, explicitly invalid depth measurements for canonical detection boxes.

Public definitions: `DepthPolicy`, `DepthEstimate`, `estimate_box_depth`, `ground_fallback`, `pinhole_fallback`, `estimate_batch_depth`.

Direct module imports: `dataclasses`, `geometry.camera`, `math`, `numpy`, `obstacle_perception.boxes`.

## `src/obstacle_perception/boxes.py`

RV-04 compatibility conversion; never invokes a detector.

Public definitions: `canonical_box`.

Direct module imports: `obstacle_perception.types`.

## `src/obstacle_perception/tracking.py`

Temporal RGB-D evidence. No output authorises clearance.

Public definitions: `TrackPolicy`, `TrackObservation`, `TrackBatch`, `TrackDepthFilter`.

Direct module imports: `collections`, `dataclasses`, `math`, `obstacle_perception.box_depth`, `obstacle_perception.types`, `statistics`.

## `src/obstacle_perception/types.py`

Sensor-derived detection and obstacle contracts (BL-C-01).

Public definitions: `ObservationMetadata`, `TrackIdentity`, `PixelBox`, `DetectionObservation`, `CallTiming`, `DetectionBatch`, `Obstacle`.

Direct module imports: `__future__`, `dataclasses`, `math`.

## `src/overlay_enhancements.py`

Showcase-quality helpers for the live perception overlay demos.

Public definitions: `LaneTemporalSmoother`, `blend_overlay`, `confidence_thickness`, `PerceptionHUD`.

Direct module imports: `__future__`, `cv2`, `dataclasses`, `numpy`, `typing`.

## `src/perception/__init__.py`

Object detection and tracking, with lazy optional learned-backend loading.

Public definitions: none at module level.

Direct module imports: `types`.

## `src/perception/frame_detector.py`

Frame contract adapter preserving image detector detect/track separation.

Public definitions: `FrameDetector`.

Direct module imports: `obstacle_perception.types`, `time`.

## `src/perception/lane_inference.py`

Loads a lane checkpoint and predicts masks from RGB frames; device selection permits CPU and supported accelerators without changing callers.

Public definitions: `LaneSegmenter`.

Direct module imports: `cv2`, `numpy`, `perception.lane_segmentation`, `torch`.

## `src/perception/lane_path.py`

Cleans lane masks and extracts ego-lane geometry; filtering and gap closing reduce isolated mask noise before path construction.

Public definitions: `extract_ego_path`, `draw_ego_path`.

Direct module imports: `cv2`, `numpy`.

## `src/perception/lane_segmentation.py`

Defines the ResNet18-based lane encoder/decoder; skip connections combine coarse context with spatial detail.

Public definitions: `DecoderBlock`, `LaneUNet`.

Direct module imports: `torch`, `torch.nn`, `torchvision.models`.

## `src/perception/object_detection.py`

Object detection and tracking for RGB images.

Public definitions: `ObjectDetector`.

Direct module imports: `numpy`, `obstacle_perception.types`, `pathlib`, `time`, `types`, `ultralytics`.

## `src/perception/recorded_detector.py`

Deterministic offline detector fixtures, bound to source frame metadata.

Public definitions: `verify_local_asset`, `RecordedDetector`, `benchmark_calls`.

Direct module imports: `hashlib`, `json`, `obstacle_perception.types`, `pathlib`, `time`.

## `src/perception/types.py`

Public types for object detection and tracking.

Public definitions: `BoundingBox`, `Detection`, `PerceptionConfig`.

Direct module imports: `dataclasses`, `math`, `obstacle_perception.types`, `typing`.

## `src/planning/__init__.py`

Routes, corridor, speed governor and behaviour (workstreams A, D and E3/E5).

Public definitions: none at module level.

Direct module imports: none.

## `src/planning/baseline_mission.py`

Load and validate one frozen, simulator-independent baseline mission.

Public definitions: `canonical_sha256`, `validate_open_polyline`, `FrozenMission`, `load_frozen_mission`.

Direct module imports: `__future__`, `dataclasses`, `hashlib`, `json`, `math`, `navigation_control`, `pathlib`, `planning.lane_graph`, `planning.route_manager`.

## `src/planning/behaviour.py`

Conservative obstacle-clearance guards for baseline behaviour (BL-D-05).

Public definitions: `ClearanceGuardConfig`, `ClearanceGuard`.

Direct module imports: `__future__`, `dataclasses`, `math`.

## `src/planning/corridor.py`

Rear-axle route coordinates and conservative swept-corridor intersection.

Public definitions: `RouteProjection`, `RouteGeometry`.

Direct module imports: `dataclasses`, `math`, `numpy`.

## `src/planning/junction_planner.py`

Legal successor traversal and exit-based maneuver classification (BL-A-03).

Public definitions: `JunctionExit`, `classify_exit`, `junction_exits`, `choose_exit`.

Direct module imports: `dataclasses`, `math`, `planning.lane_graph`.

## `src/planning/lane_graph.py`

Serializable lane sections and their explicit legal successors (BL-A-02).

Public definitions: `lane_segment_id`, `LaneSegment`, `LaneTopology`.

Direct module imports: `__future__`, `autonomy.interfaces`, `dataclasses`, `hashlib`, `json`, `math`, `pathlib`.

## `src/planning/lane_graph_audit.py`

Independent OpenDRIVE audit of Town10HD graph data, without CARLA.

Public definitions: `audit_lane_topology`.

Direct module imports: `__future__`, `collections`, `hashlib`, `math`, `planning.lane_graph`, `xml.etree.ElementTree`.

## `src/planning/route_manager.py`

Immutable lane stitching; install the resulting path once per mission.

Public definitions: `stitch_route`.

Direct module imports: `math`, `planning.junction_planner`.

## `src/planning/speed_governor.py`

Obstacle and curvature target-speed governance for BL-D-07.

Public definitions: `GovernorConfig`, `stopping_distance`, `speed_for_distance`, `SpeedGovernor`.

Direct module imports: `__future__`, `dataclasses`, `math`, `typing`.

## `src/segmentation/__init__.py`

Simulator-agnostic semantic segmentation (BiSeNetV2).

Public definitions: none at module level.

Direct module imports: `segmentation`.

## `src/segmentation/bisenetv2.py`

BiSeNetV2 architecture for real-time semantic segmentation.

Public definitions: `DetailBranch`, `StemBlock`, `GELayer`, `CEBlock`, `SemanticBranch`, `BGALayer`, `SegmentHead`, `BiSeNetV2`.

Direct module imports: `torch`, `torch.nn`.

## `src/segmentation/postprocessing.py`

Postprocessing utilities: Allocates every pixel an integer 0-18 indicating which pixel belongs

Public definitions: `colorize_mask`, `overlay_mask`.

Direct module imports: `cv2`, `numpy`.

## `src/segmentation/preprocessing.py`

Preprocessing pipeline for the semantic segmentation input frames.

Public definitions: `resize_image`, `normalize_image`, `preprocess_image`.

Direct module imports: `cv2`, `numpy`, `torch`.

## `src/segmentation/segmentation.py`

Simulator-agnostic semantic segmentation:

Public definitions: `SemanticSegmenter`.

Direct module imports: `bisenetv2`, `cv2`, `numpy`, `os`, `postprocessing`, `preprocessing`, `torch`.

## `src/sim_harness/__init__.py`

CARLA-free fixed-step world and scripted sensor fixtures (BL-F-03).

Public definitions: none at module level.

Direct module imports: `kinematic`, `synthetic`.

## `src/sim_harness/kinematic.py`

Fixed-step forward bicycle dynamics behind autonomy.interfaces.World.

Public definitions: `KinematicConfig`, `KinematicWorld`.

Direct module imports: `autonomy.interfaces`, `dataclasses`, `math`, `navigation_control.types`, `synthetic`, `time`, `uuid`.

## `src/sim_harness/s0_acceptance.py`

Offline S0 integration: shared KinematicWorld -> SkeletonStack -> world.

Public definitions: `S0MotionPolicy`, `run_s0`.

Direct module imports: `__future__`, `autonomy.interfaces`, `autonomy.stack`, `evaluation.metrics`, `evaluation.run_log`, `json`, `math`, `navigation_control`, `pathlib`, `random`, `sim_harness.kinematic`, `sim_harness.skeleton`, `sim_harness.synthetic`, `time`.

## `src/sim_harness/skeleton.py`

Contract-only fake world/detector and one-step runnable BL-F-02 evidence.

Public definitions: `EmptyDetector`, `ContractWorld`, `main`.

Direct module imports: `autonomy.interfaces`, `autonomy.stack`, `dataclasses`, `json`, `navigation_control.types`, `numpy`, `obstacle_perception.types`, `pathlib`, `time`.

## `src/sim_harness/synthetic.py`

Scripted sensor fixtures for the CARLA-free harness, never scene truth.

Public definitions: `SyntheticDetection`, `SyntheticFrame`, `SyntheticScenario`, `SyntheticDetector`, `SyntheticObstacleSource`.

Direct module imports: `autonomy.interfaces`, `dataclasses`, `geometry.camera`, `math`, `numpy`, `obstacle_perception.types`.

## `src/traffic_light_color.py`

Classify a detected traffic light's lit color from the camera crop.

Public definitions: `classify_traffic_light_color`.

Direct module imports: `cv2`, `numpy`.

## `src/traffic_perception/__init__.py`

Exports shared traffic contracts and backend entry points so callers use a stable package surface.

Public definitions: none at module level.

Direct module imports: `detector`, `distance`, `types`.

## `src/traffic_perception/_validation.py`

Validates numeric and image inputs centrally so invalid data is rejected consistently before perception work.

Public definitions: `finite`, `probability`, `positive`, `rgb_image`.

Direct module imports: `math`, `numbers`, `numpy`.

## `src/traffic_perception/detector.py`

Converts injected backend predictions into traffic detections; backend injection separates model choice from contract validation.

Public definitions: `ModelPrediction`, `TrafficPerception`.

Direct module imports: `_validation`, `collections.abc`, `dataclasses`, `distance`, `numpy`, `types`.

## `src/traffic_perception/distance.py`

Provides legacy pinhole distance and aligned-depth delegation; validity diagnostics distinguish unknown depth from clearance.

Public definitions: `estimate_distance_m`, `estimate_detection_depth`.

Direct module imports: `_validation`.

## `src/traffic_perception/types.py`

Defines traffic categories and validated detection records; the legacy box constructor reuses the canonical XYXY contract.

Public definitions: `TrafficCategory`, `BoundingBox`, `TrafficDetection`.

Direct module imports: `_validation`, `dataclasses`, `enum`, `obstacle_perception.types`.

## `src/turn_decision.py`

Junction detection and turn-direction decision logic for

Public definitions: `classify_candidate_direction`, `find_junction_ahead`, `get_junction_exit_candidates`, `select_turn_direction`, `build_turn_route`.

Direct module imports: none.

## `src/visualisation/__init__.py`

HUD, overlays, lane-departure monitor and mini-map (workstream E, E3).

Public definitions: none at module level.

Direct module imports: none.
