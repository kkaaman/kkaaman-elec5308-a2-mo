"""The car's own view while it drives (a viewer for demonstrations; no effect on the run).

Each tick, after the driver's step and outside its timing, the front camera's image as the
car received it, with what the car made of it: its own speed and target, the mode and
reason, the light it reads (state, confidence) and the decision it holds, the hold point
and the stop line, the corridor it trusts, the obstacle ahead. The band the light
classifier reads is outlined. ``--view`` shows it in a window; ``--view-video`` writes it
to ``car_view.mp4`` in the run folder at the simulation's 20 Hz, so the video plays at
the car's own pace whatever the machine's.

Only the car's own diagnostics are drawn: no evaluator truth reaches the picture.
"""

from __future__ import annotations

import math

import numpy as np

WINDOW = "Group 7: the car's view"
LIGHT_BGR = {"red": (60, 60, 230), "yellow": (40, 200, 230), "green": (80, 200, 80), "unknown": (200, 200, 200), "none": (200, 200, 200)}
# ``--view-overlays``: the segmenter's classes (ROAD_CLASSES order; "other" undrawn), in BGR.
SEGMENT_BGR = ((128, 64, 128), (0, 220, 255), (232, 35, 244), (60, 20, 220), None)
LANE_HALF_M = 1.75  # the lane drawn either side of the corridor's centre line


def _fmt(value, unit="", digits=1):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}{unit}"


def hud_lines(diagnostics, mode, reason, time_s):
    """The text drawn over the image: the car's own view of its state, line by line."""
    d = diagnostics or {}
    speed = d.get("speed_mps")
    target = d.get("target_speed_mps")
    lines = [
        f"t {time_s:5.1f} s   {_fmt(None if speed is None else speed * 3.6, ' km/h')}  (target {_fmt(None if target is None else target * 3.6, ' km/h')})",
        f"{mode}  {reason[:48]}".rstrip() + (f"   [indicator {d['indicator']}]" if d.get("indicator") else "") + (f"   [zone {d['zone_cap_mps'] * 3.6:.1f} km/h]" if d.get("zone_cap_mps") else "")
        + (f"   [overtake {d['overtake']['state']}]" if (d.get("overtake") or {}).get("state", "idle") != "idle" else ""),
    ]
    reading = d.get("light_reading")
    decision = d.get("light")
    if reading is not None or decision not in (None, "none"):
        read = "-" if reading is None else f"{reading[0]} {reading[1]:.2f}"
        lines.append(f"light: reads {read}   decision {decision or 'none'}   hold {_fmt(d.get('light_hold_m'), ' m')}   line {_fmt(d.get('stop_line_m'), ' m')}")
    lines.append(f"corridor {_fmt(d.get('corridor_valid_m'), ' m')}   obstacle {_fmt(d.get('obstacle_m'), ' m')}   cap {d.get('speed_cap_reason') or '-'}"[:96])
    return lines


class CarView:
    """Draws the front camera with the HUD; shows it (``window``) and/or records it (``video``)."""

    def __init__(self, cv2, window=True, video_path=None, camera="front", width=960, fps=20.0, crop=None, overlays=False, segment_every=2):
        self.cv2, self.window, self.video_path, self.camera, self.width, self.fps = cv2, bool(window), video_path, camera, int(width), float(fps)
        self.crop = crop  # (rows, cols) fractions of the light classifier's band, outlined
        self.overlays = bool(overlays)  # the lane, the segmenter's classes and the light reading drawn over the image
        self.segment_every = max(1, int(segment_every))
        self._labels = None  # the segmenter's last class map (the view's size)
        self.segment_note = None  # why the segmentation overlay stopped, if it did
        self.writer = None
        self.frames = 0
        self.note = None  # why the window was turned off, if it was

    def _segmentation(self, bgr, image, segmenter):
        """The segmenter's classes over the picture, every ``segment_every`` frames (the
        last map in between). The viewer runs the model itself, after the step and
        outside its timing: nothing it computes reaches the driver."""
        cv2 = self.cv2
        if segmenter is None or self.segment_note is not None:
            return bgr
        h, w = bgr.shape[:2]
        if self._labels is None or self.frames % self.segment_every == 0:
            try:
                labels = segmenter.segment(image).labels
            except Exception as error:  # the overlay is a view: the run goes on without it
                self.segment_note = f"segmentation overlay off: {error}"[:160]
                return bgr
            self._labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)
        colour = np.zeros_like(bgr)
        drawn = np.zeros((h, w), dtype=bool)
        for k, c in enumerate(SEGMENT_BGR):
            if c is not None:
                m = self._labels == k
                colour[m] = c
                drawn |= m
        mixed = cv2.addWeighted(bgr, 0.6, colour, 0.4, 0)
        bgr[drawn] = mixed[drawn]
        return bgr

    def _lane(self, bgr, image, corridor, scale, crossing):
        """The corridor the car follows, on the ground, in the picture: its centre line and
        the lane either side, as far as the corridor is valid (cyan on a junction's
        latched path, green on the kerb corridor)."""
        cv2 = self.cv2
        if corridor is None or not getattr(corridor, "ok", False):
            return bgr
        from mapless.painting import PaintingConfig, project  # noqa: PLC0415

        c = np.asarray(corridor.centre_body, dtype=float)
        far = min(40.0, float(getattr(corridor, "valid_length_m", 40.0) or 0.0))
        c = c[(c[:, 0] >= 2.0) & (c[:, 0] <= far)]
        if len(c) < 3:
            return bgr
        step = np.gradient(c, axis=0)
        norm = np.hypot(step[:, 0], step[:, 1])
        norm[norm == 0] = 1.0
        left = np.column_stack([-step[:, 1] / norm, step[:, 0] / norm])
        h0, w0 = image.rgb.shape[:2]

        def pixels(xy):
            u, v, depth, _ = project(np.column_stack([xy, np.zeros(len(xy))]), image.calibration, w0, h0, PaintingConfig())
            ok = depth > 1.0
            return np.column_stack([u[ok] * scale, v[ok] * scale]).astype(np.int32)

        centre, lp, rp = pixels(c), pixels(c + LANE_HALF_M * left), pixels(c - LANE_HALF_M * left)
        if len(lp) < 2 or len(rp) < 2:
            return bgr
        colour = (230, 200, 0) if crossing else (60, 220, 60)
        fill = bgr.copy()
        cv2.fillPoly(fill, [np.vstack([lp, rp[::-1]])], colour)
        bgr = cv2.addWeighted(fill, 0.3, bgr, 0.7, 0)
        cv2.polylines(bgr, [lp, rp], False, colour, 2, cv2.LINE_AA)
        if len(centre) >= 2:
            cv2.polylines(bgr, [centre], False, (255, 255, 255), 2, cv2.LINE_AA)
        return bgr

    def frame(self, sensor_frame, diagnostics, mode, reason, time_s, corridor=None, segmenter=None):
        """The picture for this tick (BGR), or None without the camera."""
        cv2 = self.cv2
        image = next((im for im in sensor_frame.cameras if im.name == self.camera), None)
        if image is None:
            return None
        bgr = np.ascontiguousarray(image.rgb[:, :, ::-1])
        h, w = bgr.shape[:2]
        scale = self.width / w
        bgr = cv2.resize(bgr, (self.width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = bgr.shape[:2]
        d = diagnostics or {}
        if self.overlays:
            bgr = self._segmentation(bgr, image, segmenter)
            bgr = self._lane(bgr, image, corridor, scale, bool(d.get("crossing")) and d.get("crossing") != "handed back to the kerb corridor")
        if self.crop is not None:
            (r0, r1), (c0, c1) = self.crop
            colour = LIGHT_BGR.get(d.get("light") or "none", (200, 200, 200))
            box = (int(w * c0), int(h * r0)), (int(w * c1), int(h * r1))
            cv2.rectangle(bgr, *box, colour, 3 if self.overlays else 1)
            if self.overlays:
                reading = d.get("light_reading")
                label = "light classifier: " + ("-" if reading is None else f"{str(reading[0]).upper()} {reading[1]:.2f}") + f"  -> {(d.get('light') or 'none').upper()}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                x, y = box[0][0], max(th + 6, box[0][1] - 8)
                cv2.rectangle(bgr, (x - 2, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
                cv2.putText(bgr, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
        lines = hud_lines(d, mode, reason, time_s)
        panel = 22 * len(lines) + 10
        overlay = bgr.copy()
        cv2.rectangle(overlay, (0, h - panel), (w, h), (0, 0, 0), -1)
        bgr = cv2.addWeighted(overlay, 0.55, bgr, 0.45, 0)
        for k, text in enumerate(lines):
            cv2.putText(bgr, text, (10, h - panel + 22 * (k + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        decision = d.get("light")
        if decision not in (None, "none"):
            cv2.circle(bgr, (w - 30, 30), 14, LIGHT_BGR.get(decision, (200, 200, 200)), -1)
        return bgr

    def show(self, sensor_frame, diagnostics, mode, reason, time_s, corridor=None, segmenter=None):
        if not self.window and self.video_path is None:
            return
        picture = self.frame(sensor_frame, diagnostics, mode, reason, time_s, corridor=corridor, segmenter=segmenter)
        if picture is None:
            return
        cv2 = self.cv2
        if self.video_path is not None:
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.video_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (picture.shape[1], picture.shape[0]))
            self.writer.write(picture)
        if self.window:
            try:
                cv2.imshow(WINDOW, picture)
                cv2.waitKey(1)
            except Exception as error:  # no display (a headless machine): record only
                self.window, self.note = False, f"window off: {error}"[:160]
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.window:
            try:
                self.cv2.destroyWindow(WINDOW)
            except Exception:
                pass

    def record(self):
        return dict(window=self.window, video=None if self.video_path is None else str(self.video_path), frames=self.frames, camera=self.camera, note=self.note, overlays=self.overlays, segment_note=self.segment_note)


class ChaseCamera:
    """A camera behind and above the car, its images written to a video (the CARLA window's
    ``--follow`` view, recorded without a screen recorder). Not part of the car's sensor
    suite: its images never reach the driver. If the server will not give it (an older
    blueprint, a fake world), the run goes on without it and ``note`` says why."""

    def __init__(self, carla, world, ego, path, cv2, owner_actors=None, width=1280, height=720, fps=20.0, back_m=8.0, up_m=5.0, pitch_deg=-22.0):
        self.cv2, self.path, self.fps = cv2, path, float(fps)
        self.writer, self.sensor, self.latest, self.frames, self.note = None, None, None, 0, None
        try:
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            for name, value in (("image_size_x", str(width)), ("image_size_y", str(height)), ("fov", "90"), ("sensor_tick", "0.0")):
                bp.set_attribute(name, value)
            transform = carla.Transform(carla.Location(x=-back_m, y=0.0, z=up_m), carla.Rotation(pitch=pitch_deg))
            self.sensor = world.spawn_actor(bp, transform, attach_to=ego)
            if owner_actors is not None:
                owner_actors.append(self.sensor)
            self.sensor.listen(self._receive)
        except Exception as error:
            self.sensor, self.note = None, f"chase camera off: {error}"[:160]

    def _receive(self, image):
        raw = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.latest = raw[:, :, :3].copy()  # BGRA -> BGR

    def write(self):
        """The newest chase image into the video (once a tick, after the step)."""
        if self.latest is None:
            return
        if self.writer is None:
            h, w = self.latest.shape[:2]
            self.writer = self.cv2.VideoWriter(str(self.path), self.cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(self.latest)
        self.frames += 1

    def close(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def record(self):
        return dict(video=str(self.path), frames=self.frames, note=self.note)
