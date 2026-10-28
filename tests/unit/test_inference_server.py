"""Real FastAPI routes with isolated fake tracker instances and bounded faults."""

import base64
import io
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image

from cloud.server import InferenceService, ServiceError, create_app
from perception.types import BoundingBox


class FakeDetector:
    def __init__(self):
        self.calls = 0
        self.closed = False
        self.gate = None
        self.entered = threading.Event()

    def reset(self, *, episode_id):
        self.episode_id = episode_id
        self.calls = 0

    def track(self, rgb):
        self.entered.set()
        if self.gate:
            self.gate.wait(2)
        self.calls += 1
        return (SimpleNamespace(class_name="car", confidence=.9,
                bounding_box=BoundingBox(1,1,4,4), track_id=self.calls),)

    def close(self):
        self.closed = True


def request(frame=0, episode="episode", session="a"):
    buffer = io.BytesIO()
    Image.new("RGB", (12,8)).save(buffer, format="JPEG")
    doc = dict(schema_version=1, episode_id=episode, frame_id=frame,
               sim_time_s=float(frame)*.05, image_width=12, image_height=8,
               source_id="remote", model_version="model",
               jpeg_base64=base64.b64encode(buffer.getvalue()).decode())
    headers = {"x-run-id":"run", "x-session-id":session, "x-send-monotonic-s":"10",
               "x-capture-monotonic-s":"9", "x-session-start":"true" if frame == 0 else "false"}
    return doc, headers


def service(**kwargs):
    return InferenceService(FakeDetector, model_version="model", **kwargs)


def test_two_clients_and_repeated_episode_never_share_tracker_state():
    backend = service()
    a = backend.detect(*request(session="a"))
    b = backend.detect(*request(session="b"))
    assert a["detections"][0]["track_id"] == b["detections"][0]["track_id"] == 1
    assert backend.sessions[("run","a")].detector is not backend.sessions[("run","b")].detector
    assert backend.detect(*request(1,session="a"))["detections"][0]["track_id"] == 2
    _, headers = request(session="a")
    old = backend.sessions[("run","a")].detector
    backend.reset(headers)
    assert old.closed
    assert backend.detect(*request(session="a"))["detections"][0]["track_id"] == 1
    backend.close()


@pytest.mark.parametrize("defect", ["duplicate", "old", "episode", "source", "model", "clock"])
def test_old_or_mismatched_requests_reject_without_advancing_tracker(defect):
    backend = service()
    backend.detect(*request())
    doc, headers = request(1)
    if defect == "duplicate":
        doc["frame_id"] = 0
    elif defect == "old":
        doc["sim_time_s"] = 0
    elif defect == "episode":
        doc["episode_id"] = "other"
    elif defect == "source":
        doc["source_id"] = "other"
    elif defect == "model":
        doc["model_version"] = "other"
    else:
        headers["x-capture-monotonic-s"] = "11"
    with pytest.raises(ServiceError):
        backend.detect(doc,headers)
    assert backend.sessions[("run","a")].detector.calls == 1


def test_capacity_and_expiry_require_explicit_session_reset():
    now = [0]
    backend = service(max_sessions=1, ttl_s=1, clock=lambda:now[0])
    backend.detect(*request())
    old = backend.sessions[("run","a")].detector
    with pytest.raises(ServiceError) as e:
        backend.detect(*request(session="b"))
    assert e.value.status == 429
    now[0] = 2
    with pytest.raises(ServiceError) as e:
        backend.detect(*request(1))
    assert e.value.status == 410 and old.closed
    assert backend.detect(*request(session="fresh"))["valid"]


def test_busy_session_and_reset_are_rejected_without_queueing():
    backend = service()
    backend.detect(*request())
    tracker = backend.sessions[("run","a")].detector
    tracker.entered.clear()
    tracker.gate = threading.Event()
    with ThreadPoolExecutor() as pool:
        work = pool.submit(backend.detect,*request(1))
        assert tracker.entered.wait(1)
        try:
            with pytest.raises(ServiceError) as e:
                backend.detect(*request(2))
            assert e.value.status == 409
            with pytest.raises(ServiceError):
                backend.reset(request()[1])
        finally:
            tracker.gate.set()
        assert work.result()["valid"]
    assert tracker.calls == 2


@pytest.mark.parametrize("defect", ["base64", "dimensions", "pixels", "png", "schema", "nan"])
def test_invalid_images_and_metadata_never_allocate_session(defect):
    backend = service(max_pixels=1000)
    doc, headers = request()
    if defect == "base64":
        doc["jpeg_base64"] = "!invalid"
    elif defect == "dimensions":
        doc["image_width"] = 13
    elif defect == "pixels":
        doc["image_width"] = 2000
    elif defect == "png":
        buf = io.BytesIO()
        Image.new("RGB",(12,8)).save(buf,format="PNG")
        doc["jpeg_base64"] = base64.b64encode(buf.getvalue()).decode()
    elif defect == "schema":
        doc["schema_version"] = True
    else:
        doc["sim_time_s"] = float("nan")
    with pytest.raises(ServiceError):
        backend.detect(doc,headers)
    assert not backend.sessions


def test_fastapi_auth_health_reset_and_bounded_body():
    from fastapi.testclient import TestClient

    backend = service()
    with TestClient(create_app(backend, token="test-secret",max_body_bytes=2000)) as client:
        assert client.get("/health").json()["inference_verified"] is False
        doc,headers = request()
        assert client.post("/detect",json=doc,headers=headers).status_code == 401
        headers["Authorization"] = "Bearer test-secret"
        result = client.post("/detect",json=doc,headers=headers)
        assert result.status_code == 200, result.text
        assert result.headers["x-session-id"] == "a"
        assert client.post("/detect",json=doc,headers=headers).status_code == 409
        assert client.post("/detect",content="x"*2001,headers=headers).status_code == 413
        assert client.post("/detect",content='{"frame_id":1,"frame_id":2}',headers=headers).status_code == 422
        assert client.post("/reset",headers=headers).status_code == 200
        assert not backend.sessions


def test_backend_failure_is_unavailable_and_cannot_replay_frame():
    backend = service()
    backend.detect(*request())
    tracker = backend.sessions[("run","a")].detector
    tracker.track = lambda rgb: (_ for _ in ()).throw(RuntimeError("private backend details"))
    with pytest.raises(ServiceError) as e:
        backend.detect(*request(1))
    assert e.value.status == 503 and "private" not in str(e.value)
    with pytest.raises(ServiceError) as e:
        backend.detect(*request(1))
    assert e.value.status == 503
    with pytest.raises(ServiceError) as e:
        backend.detect(*request(2))
    assert e.value.status == 503



def test_real_client_session_reset_through_fastapi_preserves_clearance_latch(monkeypatch):
    import urllib.request

    import numpy as np
    from fastapi.testclient import TestClient

    from autonomy.interfaces import CameraCalibration, EgoPose, FrameBundle
    from navigation_control.types import VehicleState
    from perception.remote_detector import JpegHttpBackend
    from planning.behaviour import ClearanceGuard

    backend = service(max_sessions=1)
    guard = ClearanceGuard()
    assert not guard.update(time_s=0, obstacle_present=True, observation_valid=True)
    with TestClient(create_app(backend,token="test-secret")) as client:
        def transport(request, timeout):
            response = client.post('/'+request.full_url.rsplit('/',1)[-1],content=request.data,
                                   headers=dict(request.header_items()))
            if response.status_code >= 400:
                raise OSError(f"HTTP {response.status_code}")
            result = io.BytesIO(response.content)
            result.headers = response.headers
            return result
        monkeypatch.setattr(urllib.request,"urlopen",transport)
        remote = JpegHttpBackend("http://test/detect",source_id="remote",model_version="model",
                                 token="test-secret")
        frame = FrameBundle("episode",0,0.0,1.0,EgoPose(VehicleState(0,0,0,0)),
                            np.zeros((8,12,3),dtype=np.uint8),
                            CameraCalibration(12,12,6,4,((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))))
        remote.reset(episode_id="episode")
        first = remote.session_id
        batch = remote.detect(frame)
        assert batch.detections[0].track_id == 1
        assert not guard.update(time_s=.05,obstacle_present=True,observation_valid=True)
        remote.reset(episode_id="episode")
        assert remote.session_id != first
        # Capacity one still permits repeated episodes: old tracker was explicitly closed.
        assert remote.detect(frame).detections[0].track_id == 1
        assert len(backend.sessions) == 1
        # A reset/ID reuse supplies no positive clearance and cannot release prior HOLD.
        assert not guard.update(time_s=.1,obstacle_present=False,observation_valid=False)
        assert guard.blocked_latch
        remote.close()
        assert not backend.sessions
