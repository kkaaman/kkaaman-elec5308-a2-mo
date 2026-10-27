"""Real loopback HTTP transport plus deterministic bounded-worker fault tests."""
import base64
import io
import json
import socket
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle
from navigation_control.types import VehicleState
from obstacle_perception.types import CallTiming, DetectionBatch, ObservationMetadata
from perception.async_detector import AsyncDetector
from perception.hybrid_detector import HybridDetector
from perception.remote_detector import JpegHttpBackend, RemoteDetector
from scripts.benchmark_remote_detector import main, summarize

_SOCKET_CONNECT = socket.socket.connect


def frame(index=0, episode="test"):
    return FrameBundle(episode, index, float(index)*.05, time.monotonic(),
        EgoPose(VehicleState(0,0,0,0)), np.zeros((8,12,3), dtype=np.uint8),
        CameraCalibration(12,12,6,4,((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))))


def batch(value, model="test-model"):
    return DetectionBatch(ObservationMetadata(value.episode_id, value.frame_id,
        value.sim_time_s, "replay", model), (),12,8,CallTiming(.001))


class Backend:
    def __init__(self, gate=None):
        self.gate = gate
        self.entered = threading.Event()
        self.calls = 0
        self.failure = False
        self.bad_identity = False
        self.closed = threading.Event()

    def reset(self, *, episode_id):
        self.episode = episode_id

    def detect(self, value):
        self.entered.set()
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(3)
        if self.failure:
            raise OSError("backend unavailable")
        result = batch(value)
        return replace(result, image_width=13) if self.bad_identity else result

    def close(self):
        self.closed.set()


@pytest.fixture
def server(monkeypatch):
    state = {"change": lambda doc: doc, "delay": 0, "requests": []}
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(request)
            image = Image.open(io.BytesIO(base64.b64decode(request.pop("jpeg_base64"))))
            assert image.size == (request["image_width"], request["image_height"])
            doc = dict(request, detections=[], valid=True, invalid_reason=None, server_duration_s=.001)
            value = state["change"](doc)
            raw = value if isinstance(value, bytes) else json.dumps(value).encode()
            time.sleep(state["delay"])
            self.send_response(200)
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass
    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=http.serve_forever, daemon=True)
    worker.start()
    state["endpoint"] = f"http://127.0.0.1:{http.server_port}/detect"
    # Override the offline guard only for this fixture's loopback address/port.
    def loopback_only(sock, address):
        if address != ("127.0.0.1", http.server_port):
            raise AssertionError("Only the mock server endpoint is allowed")
        return _SOCKET_CONNECT(sock, address)
    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    yield state
    http.shutdown()
    http.server_close()
    worker.join(1)


def remote(server, **kwargs):
    return RemoteDetector(server["endpoint"], source_id="replay", model_version="test-model", **kwargs)


def finish(worker):
    assert worker.work.done.wait(2), "Worker failed to finish"


def test_http_valid_empty_and_timing(server):
    worker = remote(server)
    try:
        worker.reset(episode_id="test")
        value = frame()
        assert worker.submit(value, deadline=time.monotonic()+1)
        finish(worker)
        assert worker.poll(value).valid
        assert worker.poll(value).detections == ()
        timing = worker.backend.last_timing
        assert timing["encode_s"] >= 0 and timing["decode_s"] >= 0
        assert timing["http_round_trip_s"] >= 0 and timing["server_duration_s"] == .001
    finally:
        worker.close()


@pytest.mark.parametrize("change", [
    lambda d: dict(d, frame_id=d["frame_id"]+1),
    lambda d: dict(d, episode_id="old"),
    lambda d: dict(d, model_version="wrong"),
    lambda d: dict(d, sim_time_s=99.0),
    lambda d: dict(d, image_width=99),
    lambda d: dict(d, frame_id=True),
    lambda d: dict(d, schema_version=2),
    lambda d: dict(d, valid="true"),
    lambda d: dict(d, unexpected=True),
    lambda d: dict(d, server_duration_s=float("nan")),
    lambda d: dict(d, detections=[{"class_name":"car","confidence":.9,"xyxy":[0,0,99,2],"track_id":None}]),
    lambda d: b'{"schema_version":1,"schema_version":1}',
    lambda d: b'not-json',
    lambda d: dict(d, valid=False, invalid_reason="no sensor"),
])
def test_http_bad_reply_is_unknown(server, change):
    server["change"] = change
    worker = remote(server)
    try:
        worker.reset(episode_id="test")
        value = frame()
        worker.submit(value, deadline=time.monotonic()+1)
        finish(worker)
        assert worker.poll(value) is None
        assert worker.work.error
    finally:
        worker.close()


def test_http_timeout_and_size_limit(server):
    server["delay"] = .08
    worker = remote(server, timeout_s=.01)
    try:
        worker.reset(episode_id="test")
        value = frame()
        worker.submit(value, deadline=time.monotonic()+1)
        finish(worker)
        assert worker.poll(value) is None and worker.work.error
    finally:
        worker.close()
    backend = JpegHttpBackend(server["endpoint"], source_id="replay", model_version="test-model",
                              max_response_bytes=4)
    backend.reset(episode_id="test")
    with pytest.raises(ValueError, match="size"):
        backend.detect(frame())


def test_nonblocking_bound_and_reset_does_not_spawn_replacements():
    gate = threading.Event()
    backend = Backend(gate)
    worker = AsyncDetector(backend)
    try:
        worker.reset(episode_id="test")
        first = frame()
        worker.submit(first, deadline=time.monotonic()+2)
        assert backend.entered.wait(1)
        start = time.monotonic()
        for index in range(1, 100):
            assert not worker.submit(frame(index), deadline=time.monotonic()+1)
            assert worker.poll(first) is None
        assert time.monotonic()-start < .5
        worker.reset(episode_id="new")
        assert not worker.submit(frame(0,"new"), deadline=time.monotonic()+1)
        gate.set()
        finish(worker)
        assert worker.poll(first) is None
        value = frame(0,"new")
        assert worker.submit(value, deadline=time.monotonic()+1)
        finish(worker)
        assert worker.poll(value).metadata.episode_id == "new"
        assert backend.calls == 2
        assert not worker.submit(value, deadline=time.monotonic()+1)
    finally:
        gate.set()
        worker.close()
        assert backend.closed.wait(1)


def test_late_completion_and_wrong_dimensions_rejected():
    clock = [0.0]
    backend = Backend()
    worker = AsyncDetector(backend, clock=lambda: clock[0])
    try:
        worker.reset(episode_id="test")
        value = frame()
        worker.submit(value, deadline=1)
        finish(worker)
        clock[0] = 2
        assert worker.poll(value) is None
        backend.bad_identity = True
        value = frame(1)
        worker.submit(value, deadline=3)
        finish(worker)
        assert worker.poll(value) is None and worker.work.error
    finally:
        worker.close()


def test_concurrent_fallback_and_distinct_frame_recovery():
    local, other = Backend(), Backend()
    remote_worker = AsyncDetector(other)
    hybrid = HybridDetector(local, remote_worker, deadline_s=1, recovery_frames=2)
    try:
        hybrid.reset(episode_id="test")
        other.failure = True
        hybrid.submit(frame())
        finish(hybrid.local)
        finish(hybrid.remote)
        assert hybrid.poll().source == "local"
        other.failure = False
        hybrid.submit(frame(1))
        finish(hybrid.local)
        finish(hybrid.remote)
        for _ in range(5):
            assert hybrid.poll().source == "local"
        hybrid.submit(frame(2))
        finish(hybrid.local)
        finish(hybrid.remote)
        assert hybrid.poll().source == "remote"
        assert local.calls == other.calls == 3
        assert not hybrid.poll().stop_required
        with pytest.raises(ValueError, match="increase"):
            hybrid.submit(frame(1))
    finally:
        hybrid.close()


def test_stalled_remote_does_not_delay_fallback_and_both_late_stop():
    gate = threading.Event()
    other = Backend(gate)
    local = Backend()
    clock = [0.0]
    worker = AsyncDetector(other, clock=lambda: clock[0])
    hybrid = HybridDetector(local, worker, deadline_s=1, clock=lambda: clock[0])
    try:
        hybrid.reset(episode_id="test")
        hybrid.submit(frame())
        finish(hybrid.local)
        assert other.entered.wait(1)
        assert hybrid.poll().source == "local"
        clock[0] = 2
        result = hybrid.poll()
        assert result.stop_required and result.batch is None
        gate.set()
        finish(worker)
        assert hybrid.poll().stop_required
        hybrid.reset(episode_id="new")
        assert hybrid.poll().stop_required
    finally:
        gate.set()
        hybrid.close()


def test_pending_and_dual_error_request_stop():
    local, other = Backend(), Backend()
    local.failure = other.failure = True
    hybrid = HybridDetector(local, AsyncDetector(other))
    try:
        hybrid.reset(episode_id="test")
        assert hybrid.poll().stop_required
        hybrid.submit(frame())
        finish(hybrid.local)
        finish(hybrid.remote)
        assert hybrid.poll().stop_required
    finally:
        hybrid.close()


def test_summary_counts_failures_in_deadline_denominator():
    result = summarize([{"status":"OK","total_s":.02},
                        {"status":"OK","total_s":.2},
                        {"status":"ERROR","total_s":.01}], .1)
    assert result["deadline_miss_rate"] == pytest.approx(2/3)
    assert result["failure_count"] == 1
    assert result["p95_s"] == pytest.approx(.191)


def test_benchmark_refuses_short_replay(tmp_path):
    with pytest.raises(SystemExit):
        main(["--frames",str(tmp_path),"--output",str(tmp_path/"out"),
              "--local-factory","missing:factory","--model-version","test-model",
              "--hardware-notes","synthetic","--origin","synthetic-smoke"])


def make_benchmark_detector():
    class ReplayBackend(Backend):
        def detect(self, value):
            return replace(batch(value), image_width=value.rgb.shape[1], image_height=value.rgb.shape[0])
    return ReplayBackend()


def test_200_frame_loopback_replay_writes_real_outputs(tmp_path, server, monkeypatch):
    import sys
    import types

    module = types.ModuleType("issue57_test_factory")
    module.make = make_benchmark_detector
    monkeypatch.setitem(sys.modules, module.__name__, module)
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(200):
        np.save(frames / f"{index:04d}.npy", np.full((8,12,3), index % 256, dtype=np.uint8))
    output = tmp_path / "evidence"
    assert main(["--frames",str(frames),"--output",str(output),
        "--local-factory","issue57_test_factory:make","--model-version","test-model",
        "--hardware-notes","synthetic fake model; loopback on test host, not LAN",
        "--origin","synthetic-smoke","--lan-endpoint",server["endpoint"],
        "--sizes","12x8","--qualities","85","--warmup","2","--deadline-s","1"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["replay"]) == 200
    assert summary["modes"]["cloud"]["status"] == "NOT RUN"
    assert summary["accuracy"]["status"] == "NOT_MEASURED"
    for mode in ("local", "LAN"):
        configuration = summary["modes"][mode]["configurations"][0]
        assert configuration["frames"] == configuration["successful_frames"] == 200
    assert len(server["requests"]) == 202  # warm-up excluded from output rows
    assert (output / "latency.png").stat().st_size > 100
    assert len((output / "latency.csv").read_text().splitlines()) == 401


def test_replay_accuracy_class_aware_one_to_one_and_rescaling():
    from obstacle_perception.types import DetectionObservation, PixelBox
    from perception.replay_accuracy import match_counts

    detection = DetectionObservation("car", .9, PixelBox(2,2,6,6))
    truth = [{"class_name":"car", "xyxy":[1,1,3,3]}]
    assert match_counts((detection,detection),truth,scale_x=2,scale_y=2) == (1,1,0)
    assert match_counts((replace(detection,class_name="person"),),truth,scale_x=2,scale_y=2) == (0,1,1)
    assert match_counts((),truth) == (0,0,1)


def test_annotated_replay_reports_missed_truth(tmp_path, monkeypatch):
    import sys
    import types

    module = types.ModuleType("issue57_annotated_factory")
    module.make = make_benchmark_detector
    monkeypatch.setitem(sys.modules, module.__name__, module)
    frames = tmp_path / "frames"
    frames.mkdir()
    labels = {}
    for index in range(200):
        name = f"{index:04d}.npy"
        np.save(frames / name, np.zeros((8,12,3), dtype=np.uint8))
        labels[name] = [{"class_name":"car","xyxy":[1,1,3,3]}]
    annotation_file = tmp_path / "truth.json"
    annotation_file.write_text(json.dumps(labels))
    output = tmp_path / "evidence"
    assert main(["--frames",str(frames),"--output",str(output),
        "--annotations",str(annotation_file),"--local-factory","issue57_annotated_factory:make",
        "--model-version","test-model","--hardware-notes","fake empty detector",
        "--origin","synthetic-smoke","--sizes","12x8","--warmup","1"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    accuracy = summary["modes"]["local"]["configurations"][0]["accuracy"]
    assert accuracy["fn"] == 200 and accuracy["recall"] == 0
    assert "NOT RUN" in (output / "comparison.md").read_text()
