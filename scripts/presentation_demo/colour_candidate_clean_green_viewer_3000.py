import json
import math
import threading
import time
from collections import deque
from pathlib import Path

import carla
import cv2
import numpy as np
from traffic_light_color_candidate import classify_traffic_light_color

import demo_combined_perception as demo

HOST = "172.28.176.1"
PORT = 3000

ROUTE_PATH = Path(
    "/home/alan/elec5308-demo-frozen/town10hd_demo_long.json"
)

STOP_TARGET_FILE = Path(
    "/home/alan/elec5308-demo-frozen/stop_target.txt"
)

STOP_DONE_FILE = Path(
    "/home/alan/elec5308-demo-frozen/stop_done.flag"
)

STOP_TARGET_FILE.unlink(missing_ok=True)
STOP_DONE_FILE.unlink(missing_ok=True)

route_raw = json.loads(ROUTE_PATH.read_text())

stop_route = [
    carla.Location(
        x=float(x),
        y=float(y),
        z=float(z),
    )
    for x, y, z in route_raw["points_xyz_m"]
]

stop_route_index = 0
stop_sign_armed = False
stop_sign_absent_frames = 0

# Do not allow the same stop sign to rearm until the vehicle has
# physically entered and then exited the associated junction.
stop_rearm_waiting = False
stop_rearm_seen_junction = False

frame_lock = threading.Lock()

latest = {
    "frame": None,
    "frame_id": None,
}


print("Loading perception models...")

segmenter = demo.SemanticSegmenter(
    weights_path=demo.SEGMENTATION_WEIGHTS,
    input_size=(512, 256),
)

detector = demo.ObjectDetector(
    demo.PerceptionConfig(
        model_name="../yolo26n.pt",
        confidence_threshold=0.20,
    )
)

lane_segmenter = demo.LaneSegmenter(
    checkpoint_path=demo.LANE_WEIGHTS,
)

shared_detection_state = {
    "detections": ()
}

traffic_perception = demo.TrafficPerception(
    demo.make_cached_backend(
        shared_detection_state
    ),
    confidence_threshold=0.4,
    object_widths_m=demo.OBJECT_WIDTHS_M,
)

cached_lane_path = {
    "value": None
}

smoother = demo.LaneTemporalSmoother()

hud = demo.PerceptionHUD(
    legend=(
        (
            "General object",
            demo.GENERAL_BOX_COLOR_BGR,
        ),
        (
            "Traffic light/sign",
            demo.TRAFFIC_BOX_COLOR_BGR,
        ),
    )
)

visible_layers = {
    "segmentation": False,
    "objects": True,
    "traffic": True,
    "lane": True,
}

print("Models loaded.")
print("Waiting for long-demo Tesla...")
print()
print("Keys:")
print("  1 = segmentation")
print("  2 = objects")
print("  3 = traffic")
print("  4 = lane")
print("  0 = all on/off")
print("  q = close viewer")
print()

client = carla.Client(HOST, PORT)
client.set_timeout(10.0)

ego = None
world = None

# Any baseline-s0 vehicle that exists when the viewer starts belongs to an
# earlier/stale demo session. Never attach the camera to one of these.
initial_world = client.get_world()

stale_ego_ids = {
    vehicle.id
    for vehicle in initial_world.get_actors().filter("vehicle.*")
    if vehicle.attributes.get("role_name") == "baseline-s0"
}

if stale_ego_ids:
    print(
        "Ignoring stale baseline-s0 actor(s):",
        sorted(stale_ego_ids),
    )

print("Waiting for NEW long-demo Tesla...")

while ego is None:
    try:
        world = client.get_world()

        for vehicle in world.get_actors().filter(
            "vehicle.*"
        ):
            if (
                vehicle.attributes.get("role_name") == "baseline-s0"
                and vehicle.id not in stale_ego_ids
            ):
                ego = vehicle
                break

    except RuntimeError:
        pass

    if ego is None:
        time.sleep(0.1)


print(
    "ATTACHED TO:",
    ego.id,
    ego.type_id,
    ego.attributes.get("role_name"),
)

carla_map = world.get_map()

camera_bp = world.get_blueprint_library().find(
    "sensor.camera.rgb"
)

camera_bp.set_attribute(
    "image_size_x",
    "800",
)
camera_bp.set_attribute(
    "image_size_y",
    "600",
)
camera_bp.set_attribute(
    "fov",
    "90",
)
camera_bp.set_attribute(
    "sensor_tick",
    "0.0",
)
camera_bp.set_attribute(
    "enable_postprocess_effects",
    "true",
)

camera_transform = carla.Transform(
    carla.Location(
        x=1.5,
        y=0.0,
        z=2.4,
    )
)

camera = world.spawn_actor(
    camera_bp,
    camera_transform,
    attach_to=ego,
    attachment_type=carla.AttachmentType.Rigid,
)


def on_frame(image):
    array = np.frombuffer(
        image.raw_data,
        dtype=np.uint8,
    )

    array = array.reshape(
        (
            image.height,
            image.width,
            4,
        )
    )

    bgr = array[:, :, :3]

    rgb = np.ascontiguousarray(
        bgr[:, :, ::-1]
    )

    with frame_lock:
        latest["frame"] = rgb
        latest["frame_id"] = image.frame


camera.listen(on_frame)


# ------------------------------------------------------------------
# PERFECT LANE HUD VERSION
#
# The real segmentation model still produces the lane pixels.
# CARLA is used only as a narrow left/right search-band selector so
# neighbouring lanes, arrows and other road paint do not contaminate the
# detector result.
# ------------------------------------------------------------------

original_extract_ego_path = demo.extract_ego_path


def project_world(
    location,
    mask_width,
    mask_height,
):
    transform = camera.get_transform()

    world_to_camera = np.asarray(
        transform.get_inverse_matrix(),
        dtype=float,
    )

    point = np.array(
        [
            location.x,
            location.y,
            location.z,
            1.0,
        ],
        dtype=float,
    )

    sensor = world_to_camera @ point

    forward = sensor[0]
    right = sensor[1]
    up = sensor[2]

    if forward <= 0.5:
        return None

    focal = 400.0

    u = (
        focal * right / forward
        + 400.0
    )

    v = (
        focal * (-up) / forward
        + 300.0
    )

    u *= mask_width / 800.0
    v *= mask_height / 600.0

    return (
        int(round(u)),
        int(round(v)),
    )


def current_lane_roi(mask):
    height, width = mask.shape

    waypoint = carla_map.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    if waypoint is None:
        return None

    waypoints = [waypoint]
    current = waypoint

    for _ in range(10):
        nxt = current.next(2.0)

        # Do not predict a route branch through a junction.
        if len(nxt) != 1:
            break

        current = nxt[0]
        waypoints.append(current)

    left_pixels = []
    right_pixels = []

    for wp in waypoints:
        centre = wp.transform.location
        rv = wp.transform.get_right_vector()

        half_width = wp.lane_width * 0.5

        left_location = carla.Location(
            x=centre.x
            - rv.x * half_width,
            y=centre.y
            - rv.y * half_width,
            z=centre.z + 0.05,
        )

        right_location = carla.Location(
            x=centre.x
            + rv.x * half_width,
            y=centre.y
            + rv.y * half_width,
            z=centre.z + 0.05,
        )

        lp = project_world(
            left_location,
            width,
            height,
        )

        rp = project_world(
            right_location,
            width,
            height,
        )

        if lp is not None:
            left_pixels.append(lp)

        if rp is not None:
            right_pixels.append(rp)

    if (
        len(left_pixels) < 3
        or len(right_pixels) < 3
    ):
        return None

    roi = np.zeros_like(
        mask,
        dtype=np.uint8,
    )

    left_line = np.asarray(
        left_pixels,
        dtype=np.int32,
    )

    right_line = np.asarray(
        right_pixels,
        dtype=np.int32,
    )

    # This was the version you described as essentially perfect.
    band_px = max(
        12,
        int(width * 0.035),
    )

    cv2.polylines(
        roi,
        [left_line],
        False,
        1,
        band_px,
    )

    cv2.polylines(
        roi,
        [right_line],
        False,
        1,
        band_px,
    )

    return roi


def map_gated_extract_ego_path(
    mask,
    number_of_points=100,
):
    if mask is None:
        return None

    roi = current_lane_roi(mask)

    if roi is None:
        # Existing temporal smoother handles the natural fade.
        return None

    gated_mask = np.where(
        roi > 0,
        mask,
        0,
    ).astype(mask.dtype)

    return original_extract_ego_path(
        gated_mask,
        number_of_points=number_of_points,
    )


demo.extract_ego_path = (
    map_gated_extract_ego_path
)


# ------------------------------------------------------------------
# STOP-LINE TARGET
#
# Perception decides whether a stop sign exists.
# The actual route being driven decides which junction entry is ahead.
# ------------------------------------------------------------------

def nearest_stop_route_index(location):
    global stop_route_index

    start = max(
        0,
        stop_route_index - 5,
    )

    end = min(
        len(stop_route),
        stop_route_index + 70,
    )

    if stop_route_index == 0:
        end = len(stop_route)

    best = min(
        range(start, end),
        key=lambda i: (
            (
                stop_route[i].x
                - location.x
            ) ** 2
            + (
                stop_route[i].y
                - location.y
            ) ** 2
        ),
    )

    stop_route_index = max(
        stop_route_index,
        best,
    )

    return stop_route_index


def route_distance_to_junction_entry(
    location,
):
    start_index = (
        nearest_stop_route_index(
            location
        )
    )

    distance = 0.0
    previous = stop_route[start_index]

    end_index = min(
        len(stop_route),
        start_index + 80,
    )

    for i in range(
        start_index + 1,
        end_index,
    ):
        point = stop_route[i]

        distance += previous.distance(
            point
        )

        waypoint = carla_map.get_waypoint(
            point,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if (
            waypoint is not None
            and waypoint.is_junction
        ):
            return distance

        previous = point

    return None


print("RGB camera attached.")
print(
    "LANE MODE: real segmentation "
    "+ ego-boundary selector"
)
print(
    "STOP MODE: perception-armed "
    "+ route stop-line target"
)

processed_frame = None
frames_processed = 0
frame_times = deque(maxlen=20)

# One lossless pending-sample log per viewer launch, shared across episodes.
TL_SAMPLE_LOG_PATH = Path(__file__).with_name(
    f"tl_confirmation_samples_{time.time_ns()}.jsonl")
print(f"TL SAMPLE LOG {TL_SAMPLE_LOG_PATH}", flush=True)


class TrafficLightHUDLock:
    """Demo display state only; no map state or vehicle control.

    Misses count processed frames, not colour samples. Without detector IDs,
    proximity is an approximation of identity, not a physical-ID guarantee.
    """
    MISS_LIMIT = 8
    CONFIRM_READINGS = 3
    HOLD_FRAMES = 30

    def __init__(self):
        self.box = None
        self.track_id = None
        self.misses = 0
        self.confirmed = None
        self.pending = None
        self.pending_count = 0
        self.last_valid_frame = -1000
        self.passed = False

    @staticmethod
    def coords(detection):
        b = detection.bounding_box
        return (float(b.x_min), float(b.y_min),
                float(b.x_max), float(b.y_max))

    @staticmethod
    def unreliable(box, shape):
        h, w = shape[:2]
        x0, y0, x1, y1 = box
        bh = y1 - y0
        # Clipping, overhead position, or excessive scale means the lamp
        # face may be disappearing. These are display guard assumptions.
        return (x0 <= 1 or y0 <= 1 or x1 >= w-1 or y1 >= h-1
                or (y0+y1)*0.5 < h*0.12
                or bh > h*0.35 or x1-x0 > w*0.20)

    def trace_sample(self, frame_number, raw=None, allowed=False,
                     reject=None, reset=None, before=None, detail=None):
        # Record scheduled decisions while unconfirmed, including the sample
        # which first confirms a colour. No second classifier evaluation.
        if ((frame_number % 3 and not (reset and before and before[1]))
                or (before is not None and before[2] is not None)):
            return
        box = self.box
        if not hasattr(self,"_trace_source_id"):
            self._trace_source_id = time.time_ns()
        record = dict(frame=frame_number, track=self.track_id,
                      source_id=self._trace_source_id,
                      source=box, raw=raw, sampling_allowed=allowed,
                      reject=reject, reset=reset, detail=detail,
                      candidate=self.pending, count=self.pending_count,
                      confirmed=self.confirmed, before=before,
                      misses=self.misses, passed=self.passed,
                      width=None if box is None else box[2]-box[0],
                      height=None if box is None else box[3]-box[1],
                      cadence=3, component_scores="not_exposed_by_classifier")
        with TL_SAMPLE_LOG_PATH.open("a") as sample_file:
            sample_file.write(json.dumps(record)+"\n")
        now = time.monotonic()
        # Lossless file, one terminal summary per second; include the raw
        # sequence since the last summary so alternation is still visible.
        self._trace_sequence = getattr(self,"_trace_sequence",[])
        self._trace_sequence.append(f"{frame_number}:{raw or '-'}:{reject or 'valid'}")
        if (now-getattr(self,"_trace_time",-float("inf")) >= 1
                or self.confirmed is not None):
            print(f"TL SAMPLE source_id={self._trace_source_id} track={self.track_id} source={box} raw={raw} "
                  f"allowed={allowed} reject={reject} reset={reset} "
                  f"candidate={self.pending} count={self.pending_count} "
                  f"confirmed={self.confirmed} size={record['width']}x{record['height']} "
                  f"sequence={','.join(self._trace_sequence)}",flush=True)
            self._trace_time = now
            self._trace_sequence = []

    @staticmethod
    def geometry_reason(box, shape):
        h,w = shape[:2]
        x0,y0,x1,y1 = box
        if x0<=1 or y0<=1 or x1>=w-1 or y1>=h-1:
            return "frame_partial"
        if (y0+y1)*.5 < h*.12:
            return "overhead_position"
        if y1-y0 > h*.35:
            return "box_height"
        if x1-x0 > w*.20:
            return "box_width"
        return None

    def update(self, detections, frame, frame_number, classifier, confirmation_only=False):
        before = (self.pending,self.pending_count,self.confirmed)
        candidates = [d for d in detections
                      if d.class_name == "traffic light"]
        selected = None
        if self.box is None:
            usable = [d for d in candidates if d.confidence >= 0.20
                      and not self.unreliable(self.coords(d), frame.shape)]
            if usable:
                selected = min(usable, key=lambda d: (
                    abs((self.coords(d)[0]+self.coords(d)[2])*0.5
                        - frame.shape[1]*0.5), -float(d.confidence)))
        else:
            if self.track_id is not None:
                selected = next((d for d in candidates
                                 if getattr(d, "track_id", None)
                                 == self.track_id), None)
            if selected is None and not (confirmation_only and self.track_id is not None):
                # In a junction, a known ID must match exactly; an anonymous
                # detection cannot replace the previously identified source.
                # Never substitute a detection bearing a different known ID.
                fallback = [d for d in candidates if self.track_id is None
                            or getattr(d, "track_id", None) is None]
                x0, y0, x1, y1 = self.box
                cx, cy = (x0+x1)*0.5, (y0+y1)*0.5
                gate = max(12.0, min(60.0, 0.75*math.hypot(x1-x0, y1-y0)))
                distances = sorted((math.hypot(
                    (self.coords(d)[0]+self.coords(d)[2])*0.5-cx,
                    (self.coords(d)[1]+self.coords(d)[3])*0.5-cy), i, d)
                    for i, d in enumerate(fallback))
                if distances and distances[0][0] <= gate:
                    # Ambiguous nearby heads count as a miss, not a switch.
                    if len(distances) == 1 or distances[1][0] > max(
                            distances[0][0]*1.5, distances[0][0]+4):
                        selected = distances[0][2]

        if selected is None:
            self.pending, self.pending_count = None, 0
            if self.box is not None:
                self.misses += 1
                self.trace_sample(frame_number, reject="source_unmatched",
                                  reset="matching_miss",before=before)
                if self.misses >= self.MISS_LIMIT:
                    self.__init__()
            return

        self.box = self.coords(selected)
        current_id = getattr(selected, "track_id", None)
        if current_id is not None:
            self.track_id = current_id
        self.misses = 0
        # Latch the overhead guard until this source disappears, so small
        # box jitter cannot re-enable underside colour updates.
        geometry = self.geometry_reason(self.box,frame.shape)
        if geometry and not self.passed:
            self._geometry_latch_reason = geometry
        self.passed = self.passed or self.unreliable(self.box, frame.shape)
        if self.passed:
            # Geometry blocks new samples, not existing pending evidence.
            self.trace_sample(frame_number,
                              reject=getattr(self,"_geometry_latch_reason","latched_hold"),
                              before=before)
            return
        if confirmation_only and self.confirmed is not None:
            # Once confirmed, retain the existing bounded hold in junctions.
            return
        if frame_number % 3:
            return
        exception = None
        try:
            raw = classifier(frame, selected.bounding_box)
            colour = str(raw).upper()
        except Exception as error:
            raw = None
            exception = type(error).__name__
            colour = "UNKNOWN"
        if colour not in ("RED", "YELLOW", "GREEN"):
            self.pending, self.pending_count = None, 0
            self.trace_sample(frame_number,raw=None,allowed=True,
                              reject="classifier_exception" if exception else "invalid_reading",
                              reset="invalid_reading",before=before,
                              detail=exception or repr(raw))
            return
        if colour == self.pending:
            self.pending_count += 1
        else:
            self.pending, self.pending_count = colour, 1
        if self.pending_count >= self.CONFIRM_READINGS:
            self.confirmed = colour
            self.last_valid_frame = frame_number
        self.trace_sample(frame_number,raw=colour,allowed=True,before=before,
                          reset="contradictory_valid_colour" if before[0] not in (None,colour) else None)

    def display_colour(self, frame_number):
        if frame_number-self.last_valid_frame <= self.HOLD_FRAMES:
            return self.confirmed
        return None


class TrafficLightApproachHUD:
    """Acquisition and junction lifecycle around the unchanged lock matcher.

    Quality gates use colour-neutral native-pixel lamp geometry. They cannot
    prove which approach a front-facing cross-traffic light controls.
    """
    ACQUIRE_FRAMES = 4

    def __init__(self):
        self.lock = TrafficLightHUDLock()
        self.proposals = []
        self.crossing = False
        self.in_junction = False
        self.was_confirmed = None
        self.episode = 1
        print("TL NEW APPROACH episode=1", flush=True)

    def __getattr__(self, name):
        return getattr(self.lock, name)

    @staticmethod
    def face_evidence(frame, detection):
        box = TrafficLightHUDLock.coords(detection)
        x0, y0, x1, y1 = map(int, box)
        w, h = x1-x0, y1-y0
        info = dict(box=box, width=w, height=h, aspect=round(h/max(1,w),3),
                    centre=((box[0]+box[2])*.5,(box[1]+box[3])*.5),
                    quality=0.0, component=None, interior=0,
                    small_edge=False, reason="ok")
        def reject(reason):
            info["reason"] = reason
            return info
        fh, fw = frame.shape[:2]
        if x0<=1 or y0<=1 or x1>=fw-1 or y1>=fh-1:
            return reject("frame_partial")
        if (y0+y1)*.5 < fh*.12:
            return reject("overhead_position")
        if h > fh*.35 or w > fw*.20:
            return reject("oversized")
        if w<4 or h<10:
            return reject("box_size")
        if not 1.15 <= h/w <= 5:
            return reject("box_aspect")
        hsv = cv2.cvtColor(frame[y0:y1,x0:x1],cv2.COLOR_RGB2HSV)
        v = hsv[:,:,2]
        mask = ((hsv[:,:,1]>=70) & (v>=max(100,.90*float(v.max())))).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask,connectivity=8)
        if n<=1: return reject("no_illuminated_component")
        order = sorted(range(1,n),key=lambda i:stats[i,4],reverse=True)
        i = order[0]
        x,y,cw,ch,area = map(int,stats[i])
        info["component"] = (x,y,cw,ch,area)
        if area<2: return reject("component_area")
        if ch>.45*h: return reject("component_height")
        if not .5 <= cw/ch <= 2: return reject("component_aspect")
        # Targeted exception: tiny YOLO boxes quantise a lower/right lamp
        # fringe onto their boundary. Keep top/left rejection and require
        # at least the existing two-pixel evidence INSIDE the crop border.
        interior = int(np.count_nonzero(labels[1:-1,1:-1]==i))
        info["interior"] = interior
        edge = x<=0 or y<=0 or x+cw>=w or y+ch>=h
        small_edge = (edge and w<=8 and h<=20 and x>0 and y>0
                      and interior>=2)
        info["small_edge"] = small_edge
        if edge and not small_edge: return reject("component_crop_edge")
        if len(order)>1 and area<1.5*stats[order[1],4]:
            return reject("component_dominance")
        fill = float(area)/(cw*ch)
        surround = v[max(0,y-1):min(h,y+ch+1),max(0,x-1):min(w,x+cw+1)]
        surround_labels = labels[max(0,y-1):min(h,y+ch+1),max(0,x-1):min(w,x+cw+1)]
        ring = surround[surround_labels!=i]
        contrast = (float(v[labels==i].mean())-float(ring.mean()))/255 if ring.size else 0
        info["contrast"] = round(contrast,3)
        if contrast<=0: return reject("component_contrast")
        info["quality"] = fill*contrast*min(cw,ch)/max(cw,ch)
        return info

    @staticmethod
    def face_quality(frame, detection):
        return TrafficLightApproachHUD.face_evidence(frame,detection)["quality"]

    def diagnose_candidates(self, records, frame_number, stage="acquisition"):
        # Wall-clock throttling also bounds output when boxes/IDs jitter.
        now = time.monotonic()
        signature = (stage,tuple((r[0],r[1]["reason"],r[2]>=self.ACQUIRE_FRAMES)
                                 for r in records))
        changed = signature != getattr(self,"_diag_signature",None)
        elapsed = now-getattr(self,"_diag_time",-float("inf"))
        if elapsed<1.0 or (not changed and elapsed<3.0): return
        self._diag_time, self._diag_signature = now, signature
        if not records:
            print(f"TL DETECTING stage={stage} candidates=0 frame={frame_number}",flush=True)
        for track,info,count,conf,match in records:
            reason = info["reason"]
            if stage!="acquisition": reason=stage
            elif reason=="ok" and count<self.ACQUIRE_FRAMES: reason="persistence"
            status = "ACCEPTABLE" if reason=="ok" else f"REJECT={reason}"
            print(f"TL CANDIDATE frame={frame_number} track={track} conf={conf:.2f} "
                  f"box={info['box']} size={info['width']}x{info['height']} "
                  f"aspect={info['aspect']} centre={info['centre']} "
                  f"quality={info['quality']:.3f} component={info['component']} "
                  f"interior={info['interior']} small_edge={info['small_edge']} "
                  f"face={info['reason']=='ok'} persist={count} match={match} {status}",flush=True)

    def stable_candidate(self, detections, frame, frame_number=0):
        proposals = []
        records = []
        used = set()
        for d in detections:
            if d.class_name != "traffic light": continue
            track = getattr(d,"track_id",None)
            info = self.face_evidence(frame,d)
            if d.confidence < .20: info["reason"] = "confidence"
            quality = info["quality"]
            if info["reason"]!="ok":
                records.append((track,info,0,float(d.confidence),"not_evaluated"))
                continue
            box = self.lock.coords(d)
            distances = []
            match_probes = []
            for j, previous in enumerate(self.proposals):
                if j in used:
                    continue
                pb, pt, count = previous
                if track is not None and pt is not None:
                    if track == pt: distances.append((0,j))
                    match_probes.append(f"ids={track}/{pt}")
                    continue
                distance = math.hypot((box[0]+box[2]-pb[0]-pb[2])*.5,
                                      (box[1]+box[3]-pb[1]-pb[3])*.5)
                gate = max(12,min(60,.75*math.hypot(pb[2]-pb[0],pb[3]-pb[1])))
                match_probes.append(f"distance={distance:.1f}/gate={gate:.1f}")
                if distance <= gate: distances.append((distance,j))
            distances.sort()
            match = None
            if distances and (len(distances)==1 or distances[1][0]>max(
                    distances[0][0]*1.5, distances[0][0]+4)):
                match = distances[0][1]
            if match is not None:
                match_reason = f"matched:distance={distances[0][0]:.1f}"
            elif distances:
                match_reason = "ambiguous"
            elif self.proposals:
                known_ids = [item[1] for item in self.proposals]
                match_reason = f"new:no_id_or_distance_match previous_ids={known_ids}"
            else:
                match_reason = "new:no_previous_credible_candidate"
            count = 1 if match is None else self.proposals[match][2]+1
            records.append((track,info,count,float(d.confidence),
                            match_reason+" probes="+str(match_probes)))
            if match is not None: used.add(match)
            proposals.append((box,track,count,d,quality))
        self.diagnose_candidates(records,frame_number)
        self.proposals = [item[:3] for item in proposals]
        stable = [item for item in proposals if item[2]>=self.ACQUIRE_FRAMES]
        if not stable: return None
        # Face quality first, forward proximity second; never rank by hue.
        best = max(stable,key=lambda item:(item[4],
            -abs((item[0][0]+item[0][2])*.5-frame.shape[1]*.5)))
        print(f"TL ACQUIRE candidate episode={self.episode} track={best[1]} "
              f"quality={best[4]:.3f} persistence={best[2]} box={best[0]} "
              f"reason=stable_credible_best_quality",flush=True)
        return best[3]

    def update(self, detections, frame, frame_number, classifier, is_junction=None):
        if self.display_colour(frame_number) is None and (
                self.lock.box is None and (is_junction is None or self.crossing)):
            stage = ("waypoint_unavailable" if is_junction is None else
                     "junction_hold" if self.crossing else "colour_confirmation")
            records = [(getattr(d,"track_id",None),self.face_evidence(frame,d),
                        0,float(d.confidence),"not_acquiring")
                       for d in detections if d.class_name=="traffic light"]
            self.diagnose_candidates(records,frame_number,stage)
        # A failed waypoint query blocks acquisition and colour updates.
        if is_junction is None:
            self.proposals = []
            before = (self.lock.pending,self.lock.pending_count,self.lock.confirmed)
            self.lock.pending, self.lock.pending_count = None, 0
            if self.lock.box is not None:
                self.lock.trace_sample(frame_number,reject="waypoint_unavailable",
                                       reset="waypoint_unavailable",before=before)
            return
        self.in_junction = bool(is_junction)
        if self.in_junction and not self.crossing:
            self.crossing = True
            self.proposals = []
            print("TL HOLD junction acquisition=blocked locked_confirmation=geometry_gated",flush=True)
        # Junction hold blocks acquisition/replacement, not confirmation of
        # a source already acquired before entry. Geometry alone latches
        # the colour-sampling hold for that source.
        if self.lock.box is None:
            if self.crossing:
                if not self.in_junction:
                    self.new_approach()
                return
            selected = self.stable_candidate(detections,frame,frame_number)
            if selected is None: return
            self.proposals = []
            # Source persistence precedes colour confirmation.
            self.lock.update([selected],frame,frame_number,classifier)
        else:
            self.lock.update(detections,frame,frame_number,classifier,
                             confirmation_only=self.crossing)
            if self.lock.box is None:
                print("TL RELEASE source reason=consecutive_misses",flush=True)
                self.was_confirmed = None
                self.proposals = []
                if self.crossing and not self.in_junction:
                    self.new_approach()
                return
        if self.lock.confirmed != self.was_confirmed:
            self.was_confirmed = self.lock.confirmed
            if self.was_confirmed:
                print(f"TL CONFIRMED {self.was_confirmed} episode={self.episode}",flush=True)

    def new_approach(self):
        self.lock = TrafficLightHUDLock()
        self.proposals = []
        self.crossing = False
        self.was_confirmed = None
        self.episode += 1
        print(f"TL NEW APPROACH episode={self.episode}",flush=True)

    def display_colour(self, frame_number):
        # No raw or pending reading is ever eligible for the large HUD.
        return self.lock.display_colour(frame_number)


hud_source_lock = TrafficLightApproachHUD()



cv2.namedWindow(
    "ELEC5308 Autonomous Driving - Town10HD",
    cv2.WINDOW_NORMAL,
)

cv2.resizeWindow(
    "ELEC5308 Autonomous Driving - Town10HD",
    1200,
    900,
)


try:
    while True:
        try:
            if not ego.is_alive:
                print(
                    "Navigation vehicle removed. "
                    "Drive finished."
                )
                break

        except RuntimeError:
            break

        with frame_lock:
            frame_rgb = (
                latest["frame"].copy()
                if latest["frame"] is not None
                else None
            )

            frame_id = latest["frame_id"]

        if (
            frame_rgb is None
            or frame_id == processed_frame
        ):
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            time.sleep(0.001)
            continue

        processed_frame = frame_id
        started = time.time()

        current_fps = (
            len(frame_times)
            / sum(frame_times)
            if (
                frame_times
                and sum(frame_times) > 0
            )
            else None
        )

        overlay = demo.build_combined_overlay(
            frame_rgb,
            segmenter,
            detector,
            lane_segmenter,
            traffic_perception,
            shared_detection_state,
            cached_lane_path,
            smoother,
            hud,
            run_lane_detection=(
                frames_processed % 2 == 0
            ),
            visible_layers=visible_layers,
            fps=current_fps,
        )

        # Large HUD only: maintain one source; per-head labels below stay
        # independent and continue to classify every current detection.
        try:
            hud_waypoint = carla_map.get_waypoint(
                ego.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            hud_in_junction = (hud_waypoint.is_junction
                               if hud_waypoint is not None else None)
        except RuntimeError:
            hud_in_junction = None
        hud_source_lock.update(
            shared_detection_state["detections"], frame_rgb,
            frames_processed, classify_traffic_light_color, hud_in_junction,
        )
        hud_colour = hud_source_lock.display_colour(frames_processed)
        if hud_colour == "GREEN":
            cv2.rectangle(overlay, (185, 75), (615, 125), (0, 0, 0), -1)
            cv2.putText(overlay, f"TRAFFIC LIGHT | {hud_colour}",
                        (215, 110), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (255, 255, 255), 2)

        # Exactly one magenta source rectangle; during short misses it
        # remains at the last matched box and is marked as held.
        if hud_source_lock.box is not None:
            x1, y1, x2, y2 = map(int, hud_source_lock.box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 4)
            source_status = "HELD" if (hud_source_lock.misses
                                       or hud_source_lock.passed) else "LOCKED"
            cv2.putText(overlay,
                        f"HUD SOURCE | {source_status} | {hud_colour or 'PENDING'}",
                        (x1, max(18, y1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 0, 255), 2)

        # Per-head colour labels: current image pixels only, display only.
        # Classify every traffic-light detection independently, including
        # heads that were not selected for the large central-source HUD.
        for detection in shared_detection_state["detections"]:
            if detection.class_name != "traffic light":
                continue

            try:
                head_colour = str(classify_traffic_light_color(
                    frame_rgb, detection.bounding_box,
                )).upper()
            except Exception:
                continue

            head_colours_bgr = {
                "RED": (0, 0, 255),
                "YELLOW": (0, 255, 255),
                "GREEN": (0, 255, 0),
            }
            if head_colour not in head_colours_bgr:
                continue

            box = detection.bounding_box
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(
                head_colour, font, 0.40, 1,
            )
            label_x = max(2, min(
                int(box.x_max) + 4,
                overlay.shape[1] - text_width - 2,
            ))
            label_y = max(text_height + 2, min(
                int((box.y_min + box.y_max) * 0.5),
                overlay.shape[0] - baseline - 2,
            ))
            # Black outline keeps small coloured text readable on scenery.
            cv2.putText(overlay, head_colour, (label_x, label_y),
                        font, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, head_colour, (label_x, label_y),
                        font, 0.40, head_colours_bgr[head_colour], 1,
                        cv2.LINE_AA)

        # --------------------------------------------------------------
        # Detect/arm stop sign.
        # --------------------------------------------------------------

        actionable_stop = None

        for detection in (
            shared_detection_state[
                "detections"
            ]
        ):
            if (
                detection.class_name
                != "stop sign"
            ):
                continue

            box = detection.bounding_box

            box_height = (
                box.y_max
                - box.y_min
            )

            if (
                detection.confidence >= 0.25
                and box_height >= 18
            ):
                actionable_stop = (
                    detection.confidence,
                    box_height,
                )
                break

        if actionable_stop is not None:
            stop_sign_absent_frames = 0

            if (
                not stop_sign_armed
                and not STOP_DONE_FILE.exists()
                and not stop_rearm_waiting
            ):
                stop_sign_armed = True

                confidence, box_height = (
                    actionable_stop
                )

                print(
                    "STOP SIGN ARMED "
                    f"confidence="
                    f"{confidence:.2f} "
                    f"height="
                    f"{box_height:.1f}px",
                    flush=True,
                )

        else:
            stop_sign_absent_frames += 1

        # Controller has completed the stop.
        if STOP_DONE_FILE.exists():
            stop_sign_armed = False
            STOP_TARGET_FILE.unlink(
                missing_ok=True
            )

            # After completing the stop, do not allow the same sign
            # to rearm merely because detection disappears for a few
            # frames. Require the vehicle to enter and then fully leave
            # the associated junction first.
            stop_rearm_waiting = True

            current_waypoint = (
                ego.get_world()
                .get_map()
                .get_waypoint(
                    ego.get_location(),
                    project_to_road=True,
                    lane_type=(
                        carla.LaneType.Driving
                    ),
                )
            )

            if (
                current_waypoint
                is not None
                and current_waypoint.is_junction
            ):
                stop_rearm_seen_junction = True

            junction_cleared = (
                stop_rearm_seen_junction
                and current_waypoint
                is not None
                and not current_waypoint.is_junction
            )

            if (
                stop_rearm_waiting
                and junction_cleared
                and stop_sign_absent_frames >= 20
            ):
                STOP_DONE_FILE.unlink(
                    missing_ok=True
                )

                stop_rearm_waiting = False
                stop_rearm_seen_junction = False
                stop_sign_absent_frames = 0

                print(
                    "STOP SIGN CLEARED AFTER JUNCTION "
                    "- REARMED",
                    flush=True,
                )

        elif stop_sign_armed:
            junction_distance = (
                route_distance_to_junction_entry(
                    ego.get_location()
                )
            )

            if junction_distance is not None:
                # route_distance_to_junction_entry() measures forward from
                # CARLA's vehicle actor origin. For stopping, the meaningful
                # reference is the front bumper: the nose must remain before
                # the stop/junction line.
                front_overhang_m = float(
                    ego.bounding_box.extent.x
                )

                stop_line_distance = (
                    junction_distance
                    - front_overhang_m
                )

                STOP_TARGET_FILE.write_text(
                    f"{stop_line_distance:.3f}\n"
                )

                cv2.rectangle(
                    overlay,
                    (145, 15),
                    (655, 65),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    overlay,
                    (
                        "STOP SIGN | LINE "
                        f"{stop_line_distance:.1f}m"
                    ),
                    (170, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            else:
                STOP_TARGET_FILE.unlink(
                    missing_ok=True
                )

        else:
            STOP_TARGET_FILE.unlink(
                missing_ok=True
            )

        velocity = ego.get_velocity()

        speed_mps = math.sqrt(
            velocity.x ** 2
            + velocity.y ** 2
            + velocity.z ** 2
        )

        speed_kmh = speed_mps * 3.6

        cv2.rectangle(
            overlay,
            (
                15,
                overlay.shape[0] - 70,
            ),
            (
                300,
                overlay.shape[0] - 15,
            ),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            overlay,
            f"SPEED  {speed_kmh:5.1f} km/h",
            (
                30,
                overlay.shape[0] - 42,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            overlay,
            "Town10HD | 900 m demo route",
            (
                30,
                overlay.shape[0] - 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.imshow(
            "ELEC5308 Autonomous Driving - Town10HD",
            overlay,
        )

        frames_processed += 1

        frame_times.append(
            time.time() - started
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key in demo.LAYER_KEYS:
            layer = demo.LAYER_KEYS[key]

            visible_layers[layer] = (
                not visible_layers[layer]
            )

            print(
                f"{layer}: "
                f"{'ON' if visible_layers[layer] else 'OFF'}"
            )

        elif key == demo.ALL_LAYERS_KEY:
            turn_on = not any(
                visible_layers.values()
            )

            for layer in visible_layers:
                visible_layers[layer] = (
                    turn_on
                )

finally:
    STOP_TARGET_FILE.unlink(
        missing_ok=True
    )

    cv2.destroyAllWindows()

    try:
        camera.stop()
    except Exception:
        pass

    try:
        if camera.is_alive:
            camera.destroy()
    except Exception:
        pass
