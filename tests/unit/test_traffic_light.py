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


def test_unsure_nones_at_the_stop_line_do_not_release_a_red():
    """showcase_n1 (PC 3, 2 Nov): stood at 468's line, ``light_v2`` read the red at
    0.45-0.6 with nones at 0.44-0.55 between; three nones in a row released the car on red
    (seeds 5308, 5311). A green or none counts towards going only at ``go_confidence``:
    the unsure nones keep the red; confident ones (``light_v1`` read the change at
    0.8-0.9) or confident greens release it."""
    f = LightFilter(LightConfig())
    r = lambda s, t, c: LightReading(s, c, t)  # noqa: E731
    assert [f.update(r("red", t, 0.7), t) for t in (0.0, 0.2)] == ["none", "red"]
    assert [f.update(r("none", t, 0.47), t) for t in (0.4, 0.6, 0.8, 1.0)] == ["red"] * 4
    assert [f.update(r("green", t, 0.55), t) for t in (1.2, 1.4, 1.6)] == ["red"] * 3
    # Confident nones after a red: the light no longer read, not a green (showcase_n2,
    # rain: light_v1 read none at 0.91-0.97 standing at the line on red). They release
    # only once they have run none_release_s; greens at once.
    nones = [f.update(r("none", t, 0.9), t, standing=True) for t in (1.8, 2.0, 2.2, 4.0, 6.6, 6.8)]
    assert nones == ["red"] * 5 + ["none"] and LightConfig().none_release_s == 5.0
    # ... and only while the car stands (final/light clear_sunrise: nones through the last
    # 30 m of the approach, and a timer started there released the car 1.5-3 s into its
    # stand, on red): 6 s of nones moving, then 4.9 s standing still hold; 5 s release.
    f.reset()
    assert [f.update(r("red", t, 0.9), t) for t in (0.0, 0.2)] == ["none", "red"]
    approach = [f.update(r("none", t, 0.9), t, standing=False) for t in (0.4, 2.0, 4.0, 6.4)]
    stand = [f.update(r("none", t, 0.9), t, standing=True) for t in (6.6, 9.0, 11.5, 11.6)]
    assert approach == ["red"] * 4 and stand == ["red", "red", "red", "none"]
    f.reset()
    assert [f.update(r("red", t, 0.7), t) for t in (0.0, 0.2)] == ["none", "red"]
    assert [f.update(r("green", t, 0.8), t) for t in (0.4, 0.6, 0.8)] == ["red", "red", "green"]


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
    assert stop_for("unknown", 10.0, 2.0) == pytest.approx(5.0) and stop_for("unknown", 1.0, 4.0) is None  # held like a red
    assert [f.update(LightReading("green", 0.9, t), t) for t in (2.0, 2.2, 2.4)] == ["unknown", "unknown", "green"]
    blind = LightFilter(LightConfig())
    assert blind.update(None, 5.0) == "none"


def test_a_red_holds_at_the_junction_entry_and_a_late_yellow_is_driven_through():
    assert stop_for("red", 20.0, 2.0) == pytest.approx(15.0)  # by default 5 m short of the sat-nav's entry (route noise up to 3.9 m)
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
    big = LightClassifier(out, device="cpu", band_scale=2.2)  # any size reads: a far light enlarged
    assert big.classify(np.zeros((72, 128, 3), dtype=np.uint8), 1.0).state in LIGHT_STATES
    from mapless.traffic_light import preprocess

    assert preprocess(np.zeros((720, 1280, 3), dtype=np.uint8), cv2, 2.2).shape == (3, 792, 1971)
    # Enlarged, the band is scaled on the device: the same input as ``preprocess`` builds
    # on the CPU, to the interpolation's rounding (showcase_n1: 15-18 ms a reading there).
    rng = np.random.default_rng(5308)
    frame = cv2.resize(rng.integers(0, 256, (90, 160, 3), dtype=np.uint8), (1280, 720), interpolation=cv2.INTER_LINEAR)
    on_device = big._input(frame)[0].numpy()
    on_cpu = preprocess(frame, cv2, 2.2)
    assert on_device.shape == on_cpu.shape and np.abs(on_device - on_cpu).mean() < 0.01 and np.abs(on_device - on_cpu).max() < 0.05
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


def test_how_a_hold_ended_is_recorded():
    """Twenty-first review (4.1): a hold ended by greens and one ended by the none timer are
    told apart (``released_by``), so a run reports every release that saw no green."""
    from mapless.traffic_light import LightConfig, LightFilter, LightReading

    f = LightFilter(LightConfig(none_release_s=1.0))
    t = 0.0
    for state in ("red", "red"):
        f.update(LightReading(state, 0.99, t), t, standing=True)
        t += 0.2
    for _ in range(3):
        f.update(LightReading("green", 0.9, t), t, standing=True)
        t += 0.2
    assert f.decision == "green" and f.released_by == "green"
    g = LightFilter(LightConfig(none_release_s=1.0))
    t = 0.0
    for state in ("red", "red"):
        g.update(LightReading(state, 0.99, t), t, standing=True)
        t += 0.2
    while g.decision == "red" and t < 5.0:
        g.update(LightReading("none", 0.95, t), t, standing=True)
        t += 0.2
    assert g.decision == "none" and g.released_by == "none timer" and t > 1.0
