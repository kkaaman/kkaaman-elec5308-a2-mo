Module notes and dataflow — DOC-05
==================================

Generated from the current ``src`` tree by ``scripts/generate_module_notes.py``.
Regenerate after adding or changing modules; ``--check`` fails on stale notes.
Roles come from source docstrings with explicit notes for older undocumented
modules. This inventory describes source responsibilities, not execution or
acceptance evidence. Public member documentation is in :doc:`api`; detailed
contract rules and review status are in :doc:`interfaces`.

Public protocol and dataflow
----------------------------

The world owns sensing and time. ``World.reset(seed)`` returns a ``FrameBundle``
and ``LaneGraph``. ``AutonomyStack.reset`` installs episode/graph state;
``AutonomyStack.step(frame)`` produces a source-bound ``ControlPacket``.
``World.step(packet)`` applies control and returns the next frame. Use the exact
signatures in :doc:`api`; protocol method presence alone does not validate
semantics. Close the world idempotently on success or failure.

``FrameBundle`` contains RGB, rear-axle pose, calibration, capture/receipt times
and optional aligned axial depth. The detector contract converts a frame to
source-bound detections; obstacle perception associates depth separately.
Planning creates paths consumed by ``navigation_control``. The current stack is
a validated skeleton; this description does not claim that every planned
perception/planning stage is integrated. Keep episode IDs, frame IDs, units and
clock domains intact. Only the adapter imports CARLA. Evaluation observations
are not perception inputs. The offline world supplies synthetic sensor fixtures,
not a validated model of collisions or live sensor performance.

Module inventory
----------------

autodrive/perception/lane_dataset.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Load paired road images and lane masks for training.

Why: Preserve inherited training/import entry points while canonical perception modules evolve.

Declared public classes/functions: ``LaneDataset``.

autodrive/perception/lane_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Legacy BGR interface. New callers should use perception.lane_inference.

Why: Preserve inherited training/import entry points while canonical perception modules evolve.

Declared public classes/functions: ``LaneSegmenter``.

autodrive/perception/lane_path.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Legacy import compatibility. Implementation lives in perception.lane_path.

Why: Preserve inherited training/import entry points while canonical perception modules evolve.

No public top-level classes/functions declared; see exports/source.

autodrive/perception/lane_segmentation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Legacy model imports. Implementation lives in perception.lane_segmentation.

Why: Preserve inherited training/import entry points while canonical perception modules evolve.

No public top-level classes/functions declared; see exports/source.

autodrive/perception/losses.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Combine segmentation losses for the inherited lane training pipeline.

Why: Preserve inherited training/import entry points while canonical perception modules evolve.

Declared public classes/functions: ``FocalLoss``, ``SoftDiceLoss``, ``CombinedLoss``.

autonomy/__init__.py
~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Contracts, the autonomy stack and safety arbitration (workstream F).

Why: Keep frame, detector and command contracts independent of the world implementation.

No public top-level classes/functions declared; see exports/source.

autonomy/admission.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Verify the versioned G1 configuration and source identity before loading.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``read_json``, ``config_identity``, ``source_inventory``, ``scope_fingerprint``, ``load_admitted_baseline``.

autonomy/baseline_operation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Load the frozen, bounded operating profile without bypassing its gate.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``frozen_governor_config``, ``baseline_operational_stack``.

autonomy/detector_contracts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    BL-C-01 protocols using the BL-F-01 frame contract.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``Detector``, ``ObstacleSource``.

autonomy/freshness.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Operational freshness gate using existing stack and command contracts.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``OperationalFreshnessGate``.

autonomy/hybrid_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Optional nonblocking hybrid detector adapter for the existing control stack.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``LocalTrackingBackend``, ``HybridFrameDetector``, ``hybrid_control_stack``.

autonomy/interfaces.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Version 1 simulator-independent packets (BL-F-01).

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``EgoPose``, ``CameraCalibration``, ``FrameBundle``, ``GraphNode``, ``GraphEdge``, ``LaneGraph``, ``Diagnostics``, ``ControlPacket``, ``World``, ``AutonomyStack``.

autonomy/lidar_fusion.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Fuse LiDAR tracks with the RGB-D corridor observation for the governor (E2-04).

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``LidarCorridorSource``, ``rgbd_report``, ``FusedObservationProvider``.

autonomy/lidar_replay.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Replay a recorded or synthetic LiDAR sweep set through the obstacle path (E2 evidence).

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``depth_report``, ``replay_manifest``, ``replay_path``.

autonomy/mission.py
~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    One fixed route installation per episode, with explicit temporary holds.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``MissionEvent``, ``MissionEpisode``.

autonomy/obstacle_policy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Freshness-aware composition of existing tracking, clearance and governance.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``CorridorObservation``, ``TrackedGovernancePolicy``.

autonomy/operational.py
~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Explicit startup entry point for freshness-gated tracked governance.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``operational_stack``.

autonomy/parking.py
~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Opt-in forward parking episode: depth -> governor -> controller -> PARK.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``ParkingEvent``, ``ParkingStep``, ``ParkingEpisode``.

autonomy/parking_contracts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Opt-in P1 contracts; A1 VehicleState and ControlCommand retain their APIs.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``finite``, ``angle_error``, ``DriveMode``, ``ParkingPhase``, ``MotionState``, ``ParkingVehicle``, ``ParkingBay``, ``ParkingGoal``, ``ParkingConfig``.

autonomy/paths.py
~~~~~~~~~~~~~~~~~

Role (source summary)::

    Repository locations that do not depend on the current working directory.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``config_path``, ``model_path``, ``run_dir``.

autonomy/stack.py
~~~~~~~~~~~~~~~~~

Role (source summary)::

    Validated, simulator-free stack skeleton with latched emergency stopping.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``StackConfig``, ``SkeletonStack``.

carla_adapter/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    CARLA integration adapter: the only package allowed to import ``carla``.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

No public top-level classes/functions declared; see exports/source.

carla_adapter/a1_capture.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Observational TEST-01 recorder around the inherited A1 demo.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``main``.

carla_adapter/adapter.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    CARLA-specific translation between CARLA and navigation_control types.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``carla_vehicle_to_state``, ``command_to_carla_control``, ``apply_command``.

carla_adapter/conversions.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Raw CARLA BGRA camera decoding without importing the simulator.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``bgra_array``, ``decode_rgb``, ``decode_depth``, ``axial_depth``.

carla_adapter/dataset_capture.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Stationary, teleported training views; never an autonomy acceptance run.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``FrameBuffer``, ``transform_dict``, ``target_surface_reference``, ``location_for_target``, ``destroy_owned_actors``, ``verify_owned_cleanup``, ``main``.

carla_adapter/depth_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Measure raw depth convention against off-axis geometry ray intersections.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``ray_reference``, ``choose_convention``, ``collect``, ``main``.

carla_adapter/environment_check.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Read-only client/server and optional GPU prerequisite check for DOC-03.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``check_versions``, ``main``.

carla_adapter/lidar.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    CARLA ray-cast LiDAR capture delivering vehicle-frame :class:`LidarSweep` packets (E2-02).

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``LidarRig``, ``lidar_points_to_ego``, ``known_object_check``, ``SweepQueue``, ``LidarCapture``.

carla_adapter/longitudinal.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    BL-E-03 throttle/brake step collection, bounded to a checked straight site.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``collect``, ``main``.

carla_adapter/s0_runner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Run one frozen, obstacle-free S0 mission and retain auditable evidence.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``configuration_digest``, ``execution_provenance``, ``signed_cross_track``, ``route_speed_target``, ``CameraReceipt``, ``RGBReceiptQueue``, ``RunSensors``, ``FrozenEnvironment``, ``run_s0_episode``, ``run_s0_suite``.

carla_adapter/scenario_verification.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Live BL-B-06/07 checks: scenario placement, evaluation truth and event sensors.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``gap_cases``, ``Verification``, ``source_provenance``, ``collect``, ``main``.

carla_adapter/scenario_world.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Live S1-S5 obstacles, evaluation truth and event sensors (BL-B-06/07).

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``core_xy``, ``route_z``, ``footprint_core``, ``ScenarioObstacle``, ``ego_front_offset``, ``ego_truth``, ``ScenarioEventSensors``, ``event_records``.

carla_adapter/sensors.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Bounded exact-frame RGB-D join; callbacks never tick the simulator.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``CameraRig``, ``AlignedCapture``, ``FrameJoiner``, ``RGBDCapture``.

carla_adapter/steering_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Short fixed-step Tesla steering sweeps through the installed vehicle adapter.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``main``.

carla_adapter/topology.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    CARLA 0.9.16 topology queries and coordinate conversion (BL-A-02).

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``TopologyError``, ``export_lane_topology``, ``validate_lane_topology_on_map``, ``connect_map``, ``load_opendrive``.

carla_adapter/vehicle.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Fixed Tesla adapter. CARLA imports stay inside this package.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``FixedVehicleAdapter``, ``main``.

carla_adapter/vehicle_geometry.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Validated fixed-vehicle geometry and simulator-free coordinate conversions.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``VehicleGeometry``, ``geometry_from_wheels``, ``forward_state``, ``normalized_steering``.

carla_adapter/weather_experiment.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Read-only installed CARLA weather/light preflight for E1-01/02.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``verify_settings``, ``select_capture_cell``, ``apply_capture_weather``, ``main``.

carla_adapter/world.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Sensor-free synchronous world ownership and cleanup for BL-B-01.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``WorldTiming``, ``SynchronousWorldSession``, ``main``.

cloud/__init__.py
~~~~~~~~~~~~~~~~~

Role (source summary)::

    Optional inference service, separate from the local baseline.

Why: Reserve the optional E4 backend namespace; its existence is not evidence of a measured extension.

No public top-level classes/functions declared; see exports/source.

cloud/server.py
~~~~~~~~~~~~~~~

Role (source summary)::

    Bounded session-isolated optional HTTP inference service (official issue #56).

Why: Reserve the optional E4 backend namespace; its existence is not evidence of a measured extension.

Declared public classes/functions: ``ServiceError``, ``Session``, ``InferenceService``, ``create_app``, ``from_environment``.

demo_combined_perception.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Combined live perception overlay: segmentation + object detection + lane detection + traffic detection, all drawn on one CARLA camera feed -- each layer individually toggleable so each module can be showcased on its own before combining them, without needing separate demo runs.

Why: Compose inherited live demonstration components at an explicit simulator entry point.

Declared public classes/functions: ``make_cached_backend``, ``draw_object_detections``, ``draw_traffic_detections``, ``draw_layer_hud``, ``build_combined_overlay``, ``measure_straightness``, ``rank_straight_spawn_candidates``, ``main``.

demo_fused_lane_following.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Fused demo: lane detection drives the vehicle during ordinary driving, handing off to a short map-based route through each junction encountered -- via navigation_control -- for turning, since pure lane-following can't decide to turn on its own.

Why: Compose inherited live demonstration components at an explicit simulator entry point.

Declared public classes/functions: ``measure_straightness``, ``rank_straight_spawn_candidates``, ``advance_past_junctions``, ``make_cached_backend``, ``draw_object_detections``, ``draw_traffic_detections``, ``draw_layer_hud``, ``main``.

evaluation/__init__.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Run logging, metrics and evaluation-only ground truth (workstreams B and F).

Why: Keep recorded observations and metric analysis separate from controller inputs.

No public top-level classes/functions declared; see exports/source.

evaluation/a1_capture.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Simulator-free TEST-01 raw logging and explicitly defined summary metrics.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``normalized_hash``, ``verify_sources``, ``signed_lane_cte``, ``metrics``, ``CaptureLog``, ``write_json``.

evaluation/a1_comparison.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    TEST-03: compare A1 and A2 navigation on one cross-track definition.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``LaneCentrelines``, ``load_a1_capture``, ``a1_metrics``, ``a2_metrics``, ``compare``.

evaluation/dataset.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Dataset contracts and metrics; no simulator or model imports.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``sha256``, ``save_json``, ``local_file``, ``road_splits``, ``decode_bgra``, ``decode_depth``, ``visible_boxes``, ``confusion``, ``metrics``, ``read_dataset``.

evaluation/longitudinal.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Measured braking budgets; absent/failed trials cannot justify a speed cap.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``braking_summary``.

evaluation/masking.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    BL-B-05 decision rule for optional semantic masking of box depth.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``masking_decision``.

evaluation/metrics.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Offline S0 metrics. Missing evidence is blocked, never interpreted as zero.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``percentile``, ``first_sustained``, ``stopped_margin``, ``read_json``, ``validate_s0_criteria``, ``evaluate_run``, ``aggregate``.

evaluation/perception_replay.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Recorded RGB-D replay; instance labels are scored after production inference.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``axial_image``, ``overlap``, ``replay``, ``range_summary``.

evaluation/run_log.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Versioned JSONL run logs; host times use one relative monotonic origin.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``plain``, ``write_json``, ``provenance``, ``RunLogger``.

evaluation/s0_metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Frozen S0 acceptance metrics with explicit evidence and failure retention.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``evaluate_s0_run``, ``aggregate_s0_runs``.

evaluation/scenario_events.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Collision and lane-invasion event records for S0-S5 (BL-B-07). Simulator-free.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``CollisionContacts``, ``collision_record``, ``lane_invasion_record``, ``validate_exclusions``, ``legal_exclusion``, ``count_contacts``, ``count_unplanned_invasions``.

evaluation/scenarios.py
~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Deterministic S1-S5 obstacle scenarios (BL-B-06). Simulator-free and evaluation-only.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``Obstacle``, ``Removal``, ``ScenarioDefinition``, ``load_scenario``, ``load_scenarios``, ``route_heading``, ``placement_pose``, ``RemovalSchedule``.

evaluation/steering_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    SI steering-response analysis; no CARLA imports or automatic adapter tuning.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``inferred_angle``, ``SpeedServo``, ``analyze``.

evaluation/stop_truth.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Evaluation-only stop-margin truth and HOLD/resume metrics (BL-B-07). Simulator-free.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``ego_front_offset_m``, ``footprint_samples``, ``signed_lateral_m``, ``route_surface_gap``, ``hold_metrics``.

evaluation/weather.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Calculate E1-03 detector results for the three approved weathers.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``evaluate_weather``, ``markdown_table``.

geometry/__init__.py
~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Camera geometry, rigid transforms and ground projection.

Why: Share explicit optical/ego/world transforms rather than duplicate coordinate assumptions.

No public top-level classes/functions declared; see exports/source.

geometry/camera.py
~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Pinhole optical rays, axial-depth projection and local ground intersection.

Why: Share explicit optical/ego/world transforms rather than duplicate coordinate assumptions.

Declared public classes/functions: ``CameraIntrinsics``, ``pixel_ray``, ``pixel_depth_to_ego``, ``ego_point_to_pixel``, ``pixel_to_ground``.

geometry/ground_projection.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    A1-compatible horizontal-camera helpers, backed by canonical geometry.

Why: Share explicit optical/ego/world transforms rather than duplicate coordinate assumptions.

Declared public classes/functions: ``pixel_to_vehicle_local``, ``project_ground_point_to_pixel``.

geometry/transforms.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Right-handed SI rigid transforms. Simulator axes are converted at its boundary.

Why: Share explicit optical/ego/world transforms rather than duplicate coordinate assumptions.

Declared public classes/functions: ``rigid_matrix``, ``transform_point``, ``inverse_rigid``, ``ego_to_world_matrix``, ``vehicle_local_to_world``.

ground_projection.py
~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Compatibility exports for the canonical geometry.ground_projection helpers.

Why: Keep the inherited helper/import interface available while reusable packages own new work.

No public top-level classes/functions declared; see exports/source.

lane_to_waypoints.py
~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Convert a lane_path.extract_ego_path() result into a short, real-world local route usable by navigation_control.PathFollower.

Why: Keep the inherited helper/import interface available while reusable packages own new work.

Declared public classes/functions: ``build_local_route``.

navigation_control/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Forward-only path following in metres, seconds and radians.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

No public top-level classes/functions declared; see exports/source.

navigation_control/_validation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Small numeric validation helpers shared by the control components.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

Declared public classes/functions: ``finite``, ``nonnegative``, ``positive``, ``clamp``.

navigation_control/controller.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Coordinate path progress, lateral control, longitudinal control and stopping.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

Declared public classes/functions: ``PathFollower``.

navigation_control/path.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Polyline validation, projection and arc-length interpolation.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

Declared public classes/functions: ``Projection``, ``ReferencePath``.

navigation_control/speed.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Discrete speed PID with conditional-integration anti-windup.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

Declared public classes/functions: ``SpeedPID``.

navigation_control/steering.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Geometric Pure Pursuit lateral controller.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

Declared public classes/functions: ``PurePursuit``.

navigation_control/types.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Immutable public input, output and configuration types.

Why: Separate path geometry, lateral control and speed control with physical-unit contracts.

Declared public classes/functions: ``VehicleState``, ``Waypoint``, ``ControllerConfig``, ``ControlStatus``, ``ControlCommand``, ``ControlDiagnostics``, ``ControlResult``.

obstacle_perception/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Obstacles from detections, depth and LiDAR (workstreams B, C and D).

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

No public top-level classes/functions declared; see exports/source.

obstacle_perception/box_depth.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Robust, explicitly invalid depth measurements for canonical detection boxes.

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``DepthPolicy``, ``DepthEstimate``, ``estimate_box_depth``, ``ground_fallback``, ``pinhole_fallback``, ``estimate_batch_depth``.

obstacle_perception/boxes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    RV-04 compatibility conversion; never invokes a detector.

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``canonical_box``.

obstacle_perception/fusion.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Conservative multi-source corridor fusion (E2-04).

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``SourceCorridorReport``, ``FusedCorridorReport``, ``fuse_corridor_reports``.

obstacle_perception/lidar.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Vehicle-frame LiDAR sweep contract (E2-01).

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``LidarConfig``, ``validate_rigid_transform``, ``LidarSweep``.

obstacle_perception/lidar_obstacles.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    LiDAR obstacles: ground removal, corridor crop, bounded clustering, tracking (E2-03/E2-04).

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``EgoBody``, ``LidarObstaclePolicy``, ``GroundSegmentation``, ``road_profile``, ``interpolate_profile``, ``remove_ground``, ``EgoRoutePose``, ``project_points``, ``corridor_mask``, ``ClusteringBudgetError``, ``cluster_points``, ``seen_clear_distance``, ``LidarCluster``, ``LidarObstacleBatch``, ``lidar_metadata``, ``footprints_overlap``, ``extract_obstacles``, ``LidarTrackObservation``, ``LidarTrackBatch``, ``LidarTrackFilter``, ``track_obstacles``.

obstacle_perception/lidar_synthetic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Deterministic synthetic LiDAR sweeps for fixtures and offline tests (E2-01/E2-03).

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``SceneBox``, ``Scene``, ``fixture_config``, ``default_sensor_to_ego``, ``ray_directions``, ``cast_sweep``, ``synthetic_sweep``, ``approach_sequence``, ``sha256_bytes``, ``build_fixture``, ``load_fixture_sweep``.

obstacle_perception/parking_clearance.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Conservative depth-buffer visibility of parking swept body prisms.

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``Clearance``, ``DepthParkingClearance``.

obstacle_perception/tracking.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Temporal RGB-D evidence. No output authorises clearance.

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``TrackPolicy``, ``TrackObservation``, ``TrackBatch``, ``TrackDepthFilter``.

obstacle_perception/types.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Sensor-derived detection and obstacle contracts (BL-C-01).

Why: Turn sensor detections/depth into explicit observations without introducing evaluation truth.

Declared public classes/functions: ``ObservationMetadata``, ``TrackIdentity``, ``PixelBox``, ``DetectionObservation``, ``CallTiming``, ``DetectionBatch``, ``Obstacle``.

overlay_enhancements.py
~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Showcase-quality helpers for the live perception overlay demos.

Why: Keep the inherited helper/import interface available while reusable packages own new work.

Declared public classes/functions: ``LaneTemporalSmoother``, ``blend_overlay``, ``confidence_thickness``, ``PerceptionHUD``.

perception/__init__.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Object detection and tracking, with lazy optional learned-backend loading.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

No public top-level classes/functions declared; see exports/source.

perception/async_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Bounded detector workers. The caller thread only submits and polls.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``positive``, ``matches``, ``Work``, ``AsyncDetector``.

perception/frame_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Frame contract adapter preserving image detector detect/track separation.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``FrameDetector``.

perception/hybrid_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    E4-05 concurrent inference with fresh same-frame local fallback.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``DetectionDecision``, ``HybridDetector``.

perception/lane_inference.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Load the local lane checkpoint and convert RGB frames into lane masks.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``LaneSegmenter``.

perception/lane_path.py
~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Extract an ego-lane path from a predicted lane mask.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``extract_ego_path``, ``draw_ego_path``.

perception/lane_segmentation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Define the lane segmentation network used by the checkpoint loader.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``DecoderBlock``, ``LaneUNet``.

perception/object_detection.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Object detection and tracking for RGB images.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``ObjectDetector``.

perception/recorded_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Deterministic offline detector fixtures, bound to source frame metadata.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``verify_local_asset``, ``RecordedDetector``, ``benchmark_calls``.

perception/remote_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    E4-04 JPEG/JSON HTTP detector; all encoding and HTTP run in a bounded worker.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``JpegHttpBackend``, ``RemoteDetector``.

perception/replay_accuracy.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Class-aware one-to-one IoU matching for explicitly annotated replay.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``match_counts``.

perception/types.py
~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Public types for object detection and tracking.

Why: Expose RGB perception and deterministic fixtures through simulator-independent APIs.

Declared public classes/functions: ``BoundingBox``, ``Detection``, ``PerceptionConfig``.

planning/__init__.py
~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Routes, corridor, speed governor and behaviour (workstreams A, D and E3/E5).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

No public top-level classes/functions declared; see exports/source.

planning/baseline_mission.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Load and validate one frozen, simulator-independent baseline mission.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``canonical_sha256``, ``validate_open_polyline``, ``FrozenMission``, ``load_frozen_mission``.

planning/behaviour.py
~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Conservative obstacle-clearance guards for baseline behaviour (BL-D-05).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``ClearanceGuardConfig``, ``ClearanceGuard``.

planning/corridor.py
~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Rear-axle route coordinates and conservative swept-corridor intersection.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``RouteProjection``, ``RouteGeometry``.

planning/junction_planner.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Legal successor traversal and exit-based maneuver classification (BL-A-03).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``JunctionExit``, ``classify_exit``, ``junction_exits``, ``choose_exit``.

planning/lane_graph.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Serializable lane sections and their explicit legal successors (BL-A-02).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``lane_segment_id``, ``LaneSegment``, ``LaneTopology``.

planning/lane_graph_audit.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Independent OpenDRIVE audit of Town10HD graph data, without CARLA.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``audit_lane_topology``.

planning/parking.py
~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Deterministic forward bay approaches with certified Bezier curvature bounds.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``ParkingSegment``, ``ParkingPlan``, ``plan_forward_parking``.

planning/parking_geometry.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Small convex geometry helpers for bounded P1 full-body sweeps.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``polygon``, ``rectangle``, ``footprint``, ``bay_polygon``, ``contains``, ``intersects``, ``swept_rectangle``.

planning/route_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Immutable lane stitching; install the resulting path once per mission.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``stitch_route``.

planning/speed_governor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Obstacle and curvature target-speed governance for BL-D-07.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``GovernorConfig``, ``stopping_distance``, ``speed_for_distance``, ``SpeedGovernor``.

planning/wayfinder/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Wayfinder v1: A* on the exported lane graph, named destinations and guidance (E3-01..03).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

No public top-level classes/functions declared; see exports/source.

planning/wayfinder/astar.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    A* over the exported ``LaneGraph`` with length cost and a planar heuristic (E3-01).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``edge_length_m``, ``euclidean_xy_m``, ``SearchResult``, ``shortest_path``.

planning/wayfinder/guidance.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Turn-by-turn guidance state: remaining distance, next manoeuvre and ETA (E3-03).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``SpeedPolicy``, ``GuidanceState``, ``RouteTracker``, ``format_distance``, ``format_eta``.

planning/wayfinder/minimap.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Route mini-map rendered with OpenCV, an existing project dependency (E3-03).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``MiniMapStyle``, ``render_minimap``, ``save_png``.

planning/wayfinder/pois.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Named points of interest snapped to legal lane-boundary nodes (E3-02).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``lane_of_node``, ``PointOfInterest``, ``PoiRegistry``.

planning/wayfinder/route.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Continuous routes between named destinations for the route manager (E3-02).

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``RoutePlanningError``, ``DestinationCheck``, ``Maneuver``, ``WayfinderRoute``, ``check_destination``, ``plan_route``, ``plan_named_route``.

segmentation/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Simulator-agnostic semantic segmentation (BiSeNetV2).

Why: Separate preprocessing, learned architecture, inference and display postprocessing.

No public top-level classes/functions declared; see exports/source.

segmentation/bisenetv2.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    BiSeNetV2 architecture for real-time semantic segmentation.

Why: Separate preprocessing, learned architecture, inference and display postprocessing.

Declared public classes/functions: ``DetailBranch``, ``StemBlock``, ``GELayer``, ``CEBlock``, ``SemanticBranch``, ``BGALayer``, ``SegmentHead``, ``BiSeNetV2``.

segmentation/postprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Postprocessing utilities: Allocates every pixel an integer 0-18 indicating which pixel belongs to which object, for example a car. Then converts it into a real RGB image by referring to the class palette. The script also blends the coloured mask into the original photo with 50% transparancy Allowing the segmentation results to be visualised.

Why: Separate preprocessing, learned architecture, inference and display postprocessing.

Declared public classes/functions: ``colorize_mask``, ``overlay_mask``.

segmentation/preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Preprocessing pipeline for the semantic segmentation input frames.

Why: Separate preprocessing, learned architecture, inference and display postprocessing.

Declared public classes/functions: ``resize_image``, ``normalize_image``, ``preprocess_image``.

segmentation/segmentation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Simulator-agnostic semantic segmentation: Combines the preprocessing, bisenetv2 as well as postprocessing scripts along with a path to the trained weights to produce the coloured overlay of the semantic segmenter

Why: Separate preprocessing, learned architecture, inference and display postprocessing.

Declared public classes/functions: ``SemanticSegmenter``.

sim_harness/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    CARLA-free fixed-step world and scripted sensor fixtures (BL-F-03).

Why: Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.

No public top-level classes/functions declared; see exports/source.

sim_harness/kinematic.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Fixed-step forward bicycle dynamics behind autonomy.interfaces.World.

Why: Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.

Declared public classes/functions: ``KinematicConfig``, ``KinematicWorld``.

sim_harness/parking.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    P1 synthetic depth rendering and independent body-contact evaluation.

Why: Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.

Declared public classes/functions: ``DepthBox``, ``render_parking_depth``, ``motion_envelope``, ``standard_case``, ``run_parking_case``.

sim_harness/s0_acceptance.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Offline S0 integration: shared KinematicWorld -> SkeletonStack -> world.

Why: Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.

Declared public classes/functions: ``S0MotionPolicy``, ``run_s0``.

sim_harness/skeleton.py
~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Contract-only fake world/detector and one-step runnable BL-F-02 evidence.

Why: Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.

Declared public classes/functions: ``EmptyDetector``, ``ContractWorld``, ``main``.

sim_harness/synthetic.py
~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Scripted sensor fixtures for the CARLA-free harness, never scene truth.

Why: Exercise world/stack contracts without CARLA; synthetic fixtures do not prove live performance.

Declared public classes/functions: ``SyntheticDetection``, ``SyntheticFrame``, ``SyntheticScenario``, ``SyntheticDetector``, ``SyntheticObstacleSource``.

traffic_light_color.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Classify a detected traffic light's lit color from the camera crop.

Why: Keep the inherited helper/import interface available while reusable packages own new work.

Declared public classes/functions: ``classify_traffic_light_color``.

traffic_perception/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Export public traffic detection, distance and configuration types.

Why: Keep traffic detections and calibrated distance calculations independent of simulator APIs.

No public top-level classes/functions declared; see exports/source.

traffic_perception/_validation.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Validate numeric and class inputs for the traffic API.

Why: Keep traffic detections and calibrated distance calculations independent of simulator APIs.

Declared public classes/functions: ``finite``, ``probability``, ``positive``, ``rgb_image``.

traffic_perception/detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Filter and normalise object detections into traffic observations.

Why: Keep traffic detections and calibrated distance calculations independent of simulator APIs.

Declared public classes/functions: ``ModelPrediction``, ``TrafficPerception``.

traffic_perception/distance.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Estimate distance from calibrated image geometry and known object dimensions.

Why: Keep traffic detections and calibrated distance calculations independent of simulator APIs.

Declared public classes/functions: ``estimate_distance_m``, ``estimate_detection_depth``.

traffic_perception/types.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Define traffic classes, bounding boxes, configuration and returned observations.

Why: Keep traffic detections and calibrated distance calculations independent of simulator APIs.

Declared public classes/functions: ``TrafficCategory``, ``BoundingBox``, ``TrafficDetection``.

turn_decision.py
~~~~~~~~~~~~~~~~

Role (source summary)::

    Junction detection and turn-direction decision logic for demo_fused_lane_following.py.

Why: Keep the inherited helper/import interface available while reusable packages own new work.

Declared public classes/functions: ``classify_candidate_direction``, ``find_junction_ahead``, ``get_junction_exit_candidates``, ``select_turn_direction``, ``build_turn_route``.

visualisation/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    HUD, overlays, lane-departure monitor and mini-map (workstream E, E3).

Why: Reserve the visualisation namespace; existing overlays still live in legacy helper modules.

No public top-level classes/functions declared; see exports/source.
