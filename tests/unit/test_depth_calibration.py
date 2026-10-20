"""Tests for installed-build depth convention selection."""

from carla_adapter.depth_calibration import choose_convention


def test_off_axis_axial_measurements_select_axial():
    rows = [
        {
            "raw_depth_m": 20.0,
            "reference": {"axial_m": 20.0, "radial_m": 25.0},
        }
        for _ in range(30)
    ]

    report = choose_convention(rows)

    assert report["status"] == "MEASURED_PASS"
    assert report["convention"] == "axial"


def test_off_axis_radial_measurements_select_radial():
    rows = [
        {
            "raw_depth_m": 25.0,
            "reference": {"axial_m": 20.0, "radial_m": 25.0},
        }
        for _ in range(30)
    ]

    report = choose_convention(rows)

    assert report["status"] == "MEASURED_PASS"
    assert report["convention"] == "radial"


def test_insufficient_or_ambiguous_measurements_are_inconclusive():
    rows = [
        {
            "raw_depth_m": 20.0,
            "reference": {"axial_m": 20.0, "radial_m": 25.0},
        }
        for _ in range(10)
    ]
    assert choose_convention(rows)["status"] == "INCONCLUSIVE"

    ambiguous = [
        {
            "raw_depth_m": 22.5,
            "reference": {"axial_m": 20.0, "radial_m": 25.0},
        }
        for _ in range(30)
    ]
    assert choose_convention(ambiguous)["status"] == "INCONCLUSIVE"