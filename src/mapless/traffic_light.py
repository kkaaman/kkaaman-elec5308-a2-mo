"""Traffic-light state from the front camera (G4 M07 and G5's learned component, 2 Nov).

A small classifier reads the front image's upper middle band and says whether a light governs the car's lane
here and, if so, its colour: ``none``, ``red``, ``yellow`` or ``green``. It is trained
on the capture runs' labels (``at_traffic_light`` and the governing light's state, from
the training towns only), so it learns which light is the car's own as well as its
colour; the governing light's truth is never an input.

``LightFilter`` turns per-frame readings into a decision the driver can act on: a stop
needs two red-or-yellow readings running at or above ``stop_confidence``; going again
needs ``go_frames`` green (or ``none``, the light passed) readings running at or above
``go_confidence`` (showcase_n1 on PC 3, 2 Nov: ``light_v2`` at the stop line read the red
at 0.45-0.6 with nones at 0.44-0.55 between, and three of those nones released the car on
red; ``light_v1`` at 2.2x read it at 0.98-1.0 and the green as none at 0.8-0.9); anything else
keeps the last decision. A reading older than ``max_age_s`` counts as none seen; once
readings have come and then none fresh for ``unknown_after_s`` (the camera or the
classifier lost), the decision is ``unknown``, held for like a red until ``go_frames``
fresh green or none readings say otherwise: a car that cannot see the light does not
drive into the junction on an old green.

The driver's use (``MaplessDriver``): a red or yellow decision with a route junction
entry ahead within ``LightConfig.stop_within_m`` holds the car at the entry, less
``stop_back_m``, measured from the car's front; a yellow it could not stop for
comfortably (``yellow_decel_mps2``), or a red seen too late to stop for at all
(``red_decel_mps2``), is driven through, as a driver would, rather than stopping in the
junction. Once the front is past the entry, no light holds it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LIGHT_STATES = ("none", "red", "yellow", "green")  # what the classifier reads
HOLDING = ("red", "yellow", "unknown")  # the filter's decisions that hold the car short of the line


@dataclass(frozen=True)
class LightConfig:
    stop_within_m: float = 30.0  # a red ahead holds the car only with a junction entry this near
    stop_back_m: float = 3.0  # without a seen stop line: the front this far short of the sat-nav's junction entry
    line_margin_m: float = 0.5  # with one (``stop_line``): the front this far short of its near edge
    line_entry_tolerance_m: float = 8.0  # a seen line is the junction's only this near its sat-nav entry (route noise: up to 3.9 m, r19s)
    camera: str = "front"
    stop_confidence: float = 0.6
    stop_frames: int = 2
    go_frames: int = 3
    go_confidence: float = 0.6  # a green or none this sure counts towards going again
    max_age_s: float = 0.6
    # With a classifier on the car: a route junction ahead is approached at approach_mps
    # at most from approach_within_m of its entry (the front's distance), so a light read
    # late can still be stopped for; and the stop line is looked for from there on.
    approach_mps: float = 3.0
    approach_within_m: float = 30.0
    unknown_after_s: float = 1.0
    yellow_decel_mps2: float = 2.0  # a yellow is stopped for only if this braking suffices
    red_decel_mps2: float = 4.0  # a red seen too late for this braking is driven through, not stopped in the box

    def __post_init__(self):
        if self.stop_frames < 1 or self.go_frames < 1:
            raise ValueError("stop_frames and go_frames must be at least 1")


@dataclass(frozen=True)
class LightReading:
    state: str
    confidence: float
    time_s: float

    def __post_init__(self):
        if self.state not in LIGHT_STATES:
            raise ValueError(f"state must be one of {LIGHT_STATES}")


class LightFilter:
    """Debounced decision from successive readings (see the module docstring)."""

    def __init__(self, config=None):
        self.config = config or LightConfig()
        self.decision = "none"
        self._stop_run = 0
        self._go_run = 0
        self.last = None

    def update(self, reading, now_s):
        c = self.config
        if reading is None or now_s - reading.time_s > c.max_age_s:
            self._stop_run = self._go_run = 0
            if self.last is not None and (reading is None or now_s - reading.time_s > c.unknown_after_s):
                self.decision = "unknown"
            return self.decision
        if reading is self.last:
            return self.decision
        self.last = reading
        stopping = reading.state in ("red", "yellow") and reading.confidence >= c.stop_confidence
        going = reading.state in ("green", "none") and reading.confidence >= c.go_confidence
        self._stop_run = self._stop_run + 1 if stopping else 0
        self._go_run = self._go_run + 1 if going else 0
        if self._stop_run >= c.stop_frames:
            self.decision = reading.state
        elif self.decision in HOLDING and self._go_run >= c.go_frames:
            self.decision = reading.state
        elif self.decision not in HOLDING and going:
            self.decision = reading.state
        return self.decision

    def reset(self):
        self.decision, self._stop_run, self._go_run, self.last = "none", 0, 0, None


def stop_for(decision, entry_m, speed_mps, config=None, back_m=None):
    """Where to hold (m ahead of the car) for this light ``decision`` and the place
    ``entry_m`` ahead to stop short of (the route junction entry, or the seen stop line),
    ``back_m`` short of it (default ``stop_back_m``), or None to drive on."""
    c = config or LightConfig()
    if decision not in HOLDING or entry_m is None or not (0.0 <= entry_m <= c.stop_within_m):
        return None
    hold = max(0.0, entry_m - (c.stop_back_m if back_m is None else back_m))
    decel = c.yellow_decel_mps2 if decision == "yellow" else c.red_decel_mps2
    if speed_mps * speed_mps > 2.0 * decel * hold:
        return None  # too close to stop (comfortably on a yellow, at all on a red): through
    return hold


# -- the classifier -------------------------------------------------------------------------
INPUT_WH = (544, 192)  # the crop below at about 0.6 scale
CROP_ROWS = (0.10, 0.60)  # of the image height
CROP_COLS = (0.15, 0.85)  # of the width


def crop(rgb):
    """The band of the front image a governing light appears in: rows 10-60 %, columns
    15-85 %. At 1280 x 720 a far-side light at a Town10HD-sized junction is ~15 x 30 px
    (live, junction 468): kept at ~0.6 scale it stays ~9 x 18 px, where a whole-image
    input at 320 px wide made it 4 x 8."""
    h, w = rgb.shape[:2]
    return rgb[int(round(h * CROP_ROWS[0])) : int(round(h * CROP_ROWS[1])), int(round(w * CROP_COLS[0])) : int(round(w * CROP_COLS[1]))]


def tiny_net(torch, classes=len(LIGHT_STATES)):
    """A small convolutional classifier (~0.3 M parameters): four conv-BN-ReLU blocks
    with stride, global pooling, a linear head. Trained from scratch on the captures."""
    nn = torch.nn

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    return nn.Sequential(block(3, 24), block(24, 48), block(48, 96), block(96, 128), nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, classes))


def preprocess(rgb, cv2, band_scale=None):
    """uint8 RGB (H x W x 3) to a float32 CHW array in [0, 1], cropped and resized: to
    ``INPUT_WH`` (as trained), or the band times ``band_scale`` (the network is fully
    convolutional with a global max pool, so any size reads: a far light enlarged)."""
    band = crop(rgb)
    if band_scale is None:
        image = cv2.resize(band, INPUT_WH, interpolation=cv2.INTER_AREA)
    else:
        h, w = band.shape[:2]
        size = (int(round(w * band_scale)), int(round(h * band_scale)))
        image = cv2.resize(band, size, interpolation=cv2.INTER_LINEAR if band_scale >= 1.0 else cv2.INTER_AREA)
    return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32) / 255.0


class LightClassifier:
    """The trained classifier, from a checkpoint ``{"state_dict", "provenance"}``."""

    def __init__(self, checkpoint_path, device=None, band_scale=None):
        import torch  # noqa: PLC0415

        self.band_scale = band_scale  # None: the trained input size (``preprocess``)

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        provenance = dict(payload.get("provenance") or {})
        if tuple(provenance.get("states", LIGHT_STATES)) != LIGHT_STATES:
            raise ValueError("light classifier trained for other states")
        self.provenance = provenance
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = tiny_net(torch)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval().to(self.device)
        self._torch = torch

    def _input(self, rgb):
        """The network's input on its device. Enlarged (``band_scale`` >= 1, the live
        reading of a far light), the band goes to the device as uint8 and is scaled and
        normalised there: at 2.2x the float image built on the CPU took 15-18 ms of the 50 ms
        tick (showcase_n1, PC 3: the light_v1 runs 6-10 % of ticks over the budget) and
        was 19 MB to copy. Same classes on the 272 recorded c1 and a1rec front frames,
        probabilities within 0.01 of ``preprocess``; 6.8 against 15.8 ms on a CPU."""
        torch = self._torch
        if self.band_scale is None or self.band_scale < 1.0:
            import cv2  # noqa: PLC0415

            return torch.from_numpy(preprocess(rgb, cv2, self.band_scale))[None].to(self.device)
        band = np.ascontiguousarray(crop(rgb))
        h, w = band.shape[:2]
        x = torch.from_numpy(band).to(self.device).permute(2, 0, 1)[None].float().div_(255.0)
        size = (int(round(h * self.band_scale)), int(round(w * self.band_scale)))
        return torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def classify(self, rgb, time_s):
        torch = self._torch
        with torch.no_grad():
            p = torch.softmax(self.model(self._input(rgb)), dim=1)[0].cpu().numpy()
        k = int(np.argmax(p))
        return LightReading(LIGHT_STATES[k], float(p[k]), float(time_s))
