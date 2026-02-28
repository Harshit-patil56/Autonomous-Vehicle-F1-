"""Tests for the throttle / brake controller."""

from __future__ import annotations

import numpy as np
import pytest

from controller.bezier import BezierCurve
from controller.throttle_brake import compute_throttle_brake
from config import CONTROLLER as _CFG


@pytest.fixture()
def straight_curve() -> BezierCurve:
    """A straight-ahead Bezier curve (zero curvature)."""
    cp = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    return BezierCurve(cp)


@pytest.fixture()
def tight_curve() -> BezierCurve:
    """A sharply curving Bezier curve (high curvature)."""
    cp = np.array([[0.0, 0.0], [8.0, 3.0], [0.0, 6.0]])
    return BezierCurve(cp)


class TestComputeThrottleBrake:
    """Tests for ``compute_throttle_brake``."""

    def test_standing_start_full_throttle(self, straight_curve: BezierCurve) -> None:
        """When speed < standing-start threshold, expect full throttle."""
        throttle, brake = compute_throttle_brake(straight_curve, speed_kph=0.0)
        assert throttle == 1.0
        assert brake == 0.0

    def test_standing_start_threshold(self, straight_curve: BezierCurve) -> None:
        """Just below standing-start threshold should still give full throttle."""
        throttle, brake = compute_throttle_brake(
            straight_curve, speed_kph=_CFG.standing_start_kph - 0.1,
        )
        assert throttle == 1.0
        assert brake == 0.0

    def test_throttle_brake_range(self, straight_curve: BezierCurve) -> None:
        """Throttle and brake must be in [0, 1]."""
        for speed in [10.0, 100.0, 200.0, 350.0]:
            throttle, brake = compute_throttle_brake(straight_curve, speed)
            assert 0.0 <= throttle <= 1.0, f"throttle={throttle} at {speed} kph"
            assert 0.0 <= brake <= 1.0, f"brake={brake} at {speed} kph"

    def test_mutual_exclusivity(self, straight_curve: BezierCurve) -> None:
        """Throttle and brake should not both be positive at the same time."""
        for speed in [20.0, 100.0, 250.0]:
            throttle, brake = compute_throttle_brake(straight_curve, speed)
            assert throttle == 0.0 or brake == 0.0

    def test_slow_on_straight_gives_throttle(
        self, straight_curve: BezierCurve,
    ) -> None:
        """On a straight with low speed, expect throttle > 0."""
        throttle, brake = compute_throttle_brake(straight_curve, speed_kph=50.0)
        assert throttle > 0.0

    def test_high_speed_on_tight_curve_gives_brake(
        self, tight_curve: BezierCurve,
    ) -> None:
        """On a tight curve with high speed, expect braking."""
        throttle, brake = compute_throttle_brake(tight_curve, speed_kph=250.0)
        assert brake > 0.0

    def test_return_type(self, straight_curve: BezierCurve) -> None:
        result = compute_throttle_brake(straight_curve, speed_kph=100.0)
        assert isinstance(result, tuple)
        assert len(result) == 2
