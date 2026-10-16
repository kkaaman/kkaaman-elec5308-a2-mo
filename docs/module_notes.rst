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

autonomy/detector_contracts.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    BL-C-01 protocols using the BL-F-01 frame contract.

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``Detector``, ``ObstacleSource``.

autonomy/interfaces.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Version 1 simulator-independent packets (BL-F-01).

Why: Keep frame, detector and command contracts independent of the world implementation.

Declared public classes/functions: ``EgoPose``, ``CameraCalibration``, ``FrameBundle``, ``GraphNode``, ``GraphEdge``, ``LaneGraph``, ``Diagnostics``, ``ControlPacket``, ``World``, ``AutonomyStack``.

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

carla_adapter/environment_check.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Read-only client/server and optional GPU prerequisite check for DOC-03.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``check_versions``, ``main``.

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

carla_adapter/world.py
~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Sensor-free synchronous world ownership and cleanup for BL-B-01.

Why: Contain simulator ownership and coordinate conversion at the boundary; keep the core reusable.

Declared public classes/functions: ``WorldTiming``, ``SynchronousWorldSession``, ``main``.

cloud/__init__.py
~~~~~~~~~~~~~~~~~

Role (source summary)::

    Optional hybrid inference backend (extension E4). Off by default.

Why: Reserve the optional E4 backend namespace; its existence is not evidence of a measured extension.

No public top-level classes/functions declared; see exports/source.

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

evaluation/steering_calibration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    SI steering-response analysis; no CARLA imports or automatic adapter tuning.

Why: Keep recorded observations and metric analysis separate from controller inputs.

Declared public classes/functions: ``inferred_angle``, ``SpeedServo``, ``analyze``.

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

planning/route_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~

Role (source summary)::

    Immutable lane stitching; install the resulting path once per mission.

Why: Use explicit graph connectivity and reusable routes rather than simulator objects in control.

Declared public classes/functions: ``stitch_route``.

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
