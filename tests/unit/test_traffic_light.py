"""The traffic-light state classifier, its debounce and the stop decision (G4 M07 / G5)."""

import json

import numpy as np
import pytest

from mapless.traffic_light import LIGHT_STATES, LightConfig, LightFilter, LightReading, stop_for


def test_the_filter_needs_two_reds_to_stop_and_three_greens_to_go():
    f = LightFilter(LightConfig())
    r = lambda s, t, c=0.9: LightReading(s, c, t)  # noqa: E731
    assert f.update(r("red", 0.0), 0.0) == "none"  # one red is a glimpse
    assert f.update(r("red", 0.1), 0.1) == "red"
    assert f.update(r("green", 0.2), 0.2) == "red"  # one green does not release
    assert f.update(r("green", 0.3), 0.3) == "red"
    assert f.update(r("green", 0.4), 0.4) == "green"
    assert f.update(r("red", 0.5, c=0.4), 0.5) == "green"  # an unsure red never stops the car
    assert f.update(r("red", 0.6, c=0.4), 0.6) == "green"
    stale = r("red", 0.7)
    assert f.update(stale, 0.7) == "green" and f.update(stale, 0.75) == "green"  # the same reading counts once
    assert f.update(r("red", 0.8), 1.5) == "green"  # too old to count (0.7 s), not yet lost
    assert f.update(r("red", 0.8), 2.0) == "unknown"  # 1.2 s without a fresh reading: lost
    with pytest.raises(ValueError):
        LightReading("purple", 1.0, 0.0)


def test_a_light_the_car_has_lost_sight_of_holds_it_until_fresh_readings_release_it():
    """Eighteenth review (4.2/4.3): readings that stop coming (the camera or classifier
    lost) are not an old green to drive on: after unknown_after_s the decision is
    unknown, held for like a red; three fresh greens release it. A filter that never had
    a reading (no classifier) never goes unknown."""
    f = LightFilter(LightConfig())
    green = LightReading("green", 0.9, 0.0)
    assert f.update(green, 0.0) == "green"
    assert f.update(green, 0.5) == "green" and f.update(green, 0.9) == "green"  # stale, not yet lost
    assert f.update(green, 1.1) == "unknown"
    assert stop_for("unknown", 10.0, 2.0) == pytest.approx(7.0) and stop_for("unknown", 1.0, 4.0) is None  # held like a red
    assert [f.update(LightReading("green", 0.9, t), t) for t in (2.0, 2.2, 2.4)] == ["unknown", "unknown", "green"]
    blind = LightFilter(LightConfig())
    assert blind.update(None, 5.0) == "none"


def test_a_red_holds_at_the_junction_entry_and_a_late_yellow_is_driven_through():
    assert stop_for("red", 20.0, 2.0) == pytest.approx(17.0)  # by default 3 m short of the sat-nav's entry
    assert stop_for("red", 20.0, 2.0, back_m=0.5) == pytest.approx(19.5)  # a seen stop line: 0.5 m short of it
    c = LightConfig(stop_back_m=1.0)
    assert stop_for("red", 20.0, 2.0, c) == pytest.approx(19.0)  # the stop line, 1 m short of the entry
    assert stop_for("red", 45.0, 2.0, c) is None  # no junction near: not this light's business
    assert stop_for("red", None, 2.0, c) is None
    assert stop_for("green", 10.0, 2.0, c) is None
    assert stop_for("yellow", 10.0, 2.0, c) == pytest.approx(9.0)  # 2 m/s stops in 1 m
    assert stop_for("yellow", 2.5, 4.0, c) is None  # 4 m/s needs 4 m at 2 m/s^2: through
    assert stop_for("red", 3.5, 4.0, c) == pytest.approx(2.5)  # a red: 4 m/s stops in 2 m at 4 m/s^2
    assert stop_for("red", 2.5, 4.0, c) is None  # 1.5 m to the line: too late even braking hard, through, not stopped in the box
    assert stop_for("red", 1.0, 0.0, c) == pytest.approx(0.0)  # standing at the line: stays there
    assert stop_for("red", -0.5, 0.0, c) is None  # the front past the entry


def test_training_on_a_tiny_capture_labels_from_truth_and_refuses_test_towns(tmp_path):
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    from mapless_training import train_traffic_light as tl

    def capture(name, town, rows):
        run = tmp_path / name
        (run / "rgb" / "front").mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps(dict(map=f"Carla/Maps/{town}", status="COMPLETE")), encoding="utf-8")
        with open(run / "samples.jsonl", "w", encoding="utf-8") as out:
            for i, (at, state) in enumerate(rows):
                image = np.zeros((72, 128, 3), dtype=np.uint8)
                if at:  # a lit lamp, top right
                    colour = dict(Red=(0, 0, 255), Yellow=(0, 200, 255), Green=(0, 255, 0))[state]
                    cv2.circle(image, (100, 12), 6, colour, -1)
                cv2.imwrite(str(run / "rgb" / "front" / f"{i:08d}.jpg"), image)
                out.write(json.dumps(dict(frame=i, files=dict(rgb=dict(front=f"rgb/front/{i:08d}.jpg")), truth=dict(at_traffic_light=at, traffic_light_state=state))) + "\n")
        return run

    rows = [(False, "Green"), (True, "Red"), (True, "Green"), (True, "Yellow"), (False, "Red")] * 4
    train = capture("capture_Town03_s1", "Town03", rows)
    val = capture("capture_Town02_s1", "Town02", rows)
    items, counts = tl.index([train, val])
    assert counts[str(train)] == dict(town="Town03", approach_labelled=0, none=8, red=4, yellow=4, green=4)  # not at a light: none, whatever the state
    out = tmp_path / "light.pt"
    tl.main(["--run", str(train), "--run", str(val), "--output", str(out), "--epochs", "1", "--steps-per-epoch", "2", "--batch", "8", "--workers", "0", "--device", "cpu", "--select", "stop"])
    payload = torch.load(out, map_location="cpu", weights_only=False)
    p = payload["provenance"]
    assert p["training_towns"] == ["Town03"] and p["validation_towns"] == ["Town02"] and p["states"] == list(LIGHT_STATES)
    assert p["selection"] == "validation red recall less false-red share"
    assert "Town10HD" in p["refused_towns"] and (tmp_path / "light.provenance.json").exists()
    from mapless.traffic_light import LightClassifier

    reading = LightClassifier(out, device="cpu").classify(np.zeros((72, 128, 3), dtype=np.uint8), 1.0)
    assert reading.state in LIGHT_STATES and 0.0 <= reading.confidence <= 1.0
    import pickle

    sample = pickle.loads(pickle.dumps(tl.LightImages(items[:2], True)))  # Windows spawns DataLoader workers: the dataset must pickle
    assert len(sample) == 2 and tuple(sample[0][0].shape) == (3, 192, 544)  # augmented: zoomed back to the input size
    assert tuple(tl.LightImages(items[:2], False)[1][0].shape) == (3, 192, 544)
    held_out = capture("capture_Town10HD_s1", "Town10HD", rows)
    with pytest.raises(ValueError):
        tl.index([held_out])


def test_scores_count_false_stops_and_red_recall():
    from mapless_training.train_traffic_light import scores

    #           pred: none red yellow green
    confusion = [[90, 10, 0, 0],  # none
                 [2, 18, 0, 0],  # red
                 [0, 1, 4, 0],  # yellow
                 [0, 5, 0, 15]]  # green
    s = scores(confusion)
    assert s["red_recall"] == 0.9 and s["false_red_share"] == round(15 / 120, 4)
    assert s["green_as_red"] == 0.25 and s["red_as_green"] == 0.0
    assert s["balanced_accuracy"] == round((0.9 + 0.9 + 0.8 + 0.75) / 4, 4)


def test_the_approach_to_a_light_takes_the_state_found_at_its_stop_line():
    """The trigger zone starts near the stop line: the 20 m and 3 s before it are labelled
    with the state the car finds there, so the light is read with room to stop; a jump in
    position (a new episode) or time ends the look-back."""
    from mapless_training.train_traffic_light import approach_labels

    def row(t, x, at=False, state="Green"):
        return dict(sim_time_s=t, truth=dict(at_traffic_light=at, traffic_light_state=state, ego_rear_axle=dict(x_m=x, y_m=0.0)))

    rows = [row(0.0, 0.0), row(0.5, 4.0), row(1.0, 8.0), row(1.5, 12.0), row(2.0, 14.0, True, "Red"), row(2.5, 14.0, True, "Red")]
    assert approach_labels(rows) == ["red", "red", "red", "red", "red", "red"]
    assert approach_labels(rows, approach_m=9.0) == ["none", "none", "red", "red", "red", "red"]  # 10 m back is too far
    assert approach_labels(rows, approach_s=1.2) == ["none", "none", "red", "red", "red", "red"]  # 1.5 s back is too long
    jump = [row(0.0, 100.0)] + rows[1:]
    assert approach_labels(jump)[0] == "none"  # a teleport: another episode
    passing = [row(0.0, 0.0), row(0.5, 4.0, True, "Green"), row(1.0, 8.0)]
    assert approach_labels(passing) == ["green", "green", "none"]
