"""Tests for the Pure Pursuit steering controller."""

from __future__ import annotations

import math

import numpy as np
import pytest

from controller.bezier import BezierCurve
from controller.pure_pursuit import compute_lookahead, pure_pursuit_steer
from config import CONTROLLER as _CFG


class TestComputeLookahead:
    """Tests for ``compute_lookahead``."""

    def test_zero_speed(self) -> None:
        """At zero speed, lookahead should equal the minimum."""
        ld = compute_lookahead(0.0)
        assert ld == pytest.approx(_CFG.lookahead_min_m)

    def test_increases_with_speed(self) -> None:
        """Lookahead should increase as speed increases."""
        ld_slow = compute_lookahead(10.0)
        ld_fast = compute_lookahead(50.0)
        assert ld_fast > ld_slow

    def test_clamped_to_max(self) -> None:
        """At very high speed, lookahead must not exceed the maximum."""
        ld = compute_lookahead(500.0)  # Very high speed.
        assert ld <= _CFG.lookahead_max_m

    def test_clamped_to_min(self) -> None:
        """At any speed, lookahead must not go below the minimum."""
        ld = compute_lookahead(-10.0)  # Nonsensical but boundary check.
        assert ld >= _CFG.lookahead_min_m

    def test_return_type(self) -> None:
        ld = compute_lookahead(30.0)
        assert isinstance(ld, float)


class TestPurePursuitSteer:
    """Tests for ``pure_pursuit_steer``."""

    @pytest.fixture()
    def straight_ahead_curve(self) -> BezierCurve:
        """Curve going straight forward (positive Z, x=0)."""
        cp = np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
        return BezierCurve(cp)

    @pytest.fixture()
    def left_curve(self) -> BezierCurve:
        """Curve curving to the left (negative X)."""
        cp = np.array([[0.0, 0.0], [-5.0, 10.0], [-10.0, 20.0]])
        return BezierCurve(cp)

    @pytest.fixture()
    def right_curve(self) -> BezierCurve:
        """Curve curving to the right (positive X)."""
        cp = np.array([[0.0, 0.0], [5.0, 10.0], [10.0, 20.0]])
        return BezierCurve(cp)

    def test_straight_produces_near_zero_steer(
        self, straight_ahead_curve: BezierCurve,
    ) -> None:
        """Driving straight at moderate speed -> steer ~= 0."""
        steer = pure_pursuit_steer(straight_ahead_curve, speed_kph=100.0)
        assert abs(steer) < 0.05

    def test_left_curve_negative_steer(
        self, left_curve: BezierCurve,
    ) -> None:
        """A left-bending path should produce negative steering."""
        steer = pure_pursuit_steer(left_curve, speed_kph=80.0)
        assert steer < 0.0

    def test_right_curve_positive_steer(
        self, right_curve: BezierCurve,
    ) -> None:
        """A right-bending path should produce positive steering."""
        steer = pure_pursuit_steer(right_curve, speed_kph=80.0)
        assert steer > 0.0

    def test_output_range(self, left_curve: BezierCurve) -> None:
        """Steer must be in [-1, 1]."""
        steer = pure_pursuit_steer(left_curve, speed_kph=200.0)
        assert -1.0 <= steer <= 1.0

    def test_symmetric(
        self,
        left_curve: BezierCurve,
        right_curve: BezierCurve,
    ) -> None:
        """Left and right curves of equal magnitude should give
        approximately symmetric steering values."""
        s_left = pure_pursuit_steer(left_curve, speed_kph=100.0)
        s_right = pure_pursuit_steer(right_curve, speed_kph=100.0)
        assert abs(s_left + s_right) < 0.1  # Should be close to equal-opposite.
