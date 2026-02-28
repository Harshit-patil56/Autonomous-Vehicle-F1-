"""Tests for Bezier curve utilities."""

from __future__ import annotations

import numpy as np
import pytest

from controller.bezier import (
    BezierCurve,
    bezier_matrix,
    curvature_at,
    evaluate_bezier,
    fit_bezier_lsq,
)


class TestBezierMatrix:
    """Tests for ``bezier_matrix``."""

    def test_shape(self) -> None:
        """Output shape should be (S, order+1)."""
        t = np.linspace(0, 1, 50)
        m = bezier_matrix(order=8, t=t)
        assert m.shape == (50, 9)

    def test_partition_of_unity(self) -> None:
        """Bernstein basis polynomials must sum to 1 for all t."""
        t = np.linspace(0, 1, 200)
        m = bezier_matrix(order=5, t=t)
        np.testing.assert_allclose(m.sum(axis=1), 1.0, atol=1e-12)

    def test_endpoints(self) -> None:
        """At t=0 only B_{0,n}=1; at t=1 only B_{n,n}=1."""
        t = np.array([0.0, 1.0])
        m = bezier_matrix(order=3, t=t)
        # t=0: first basis = 1, rest = 0.
        np.testing.assert_allclose(m[0], [1, 0, 0, 0], atol=1e-14)
        # t=1: last basis = 1, rest = 0.
        np.testing.assert_allclose(m[1], [0, 0, 0, 1], atol=1e-14)

    def test_order_zero(self) -> None:
        """Order-0 curve is a single point; basis is always 1."""
        t = np.array([0.0, 0.5, 1.0])
        m = bezier_matrix(order=0, t=t)
        np.testing.assert_allclose(m, np.ones((3, 1)), atol=1e-14)


class TestEvaluateBezier:
    """Tests for ``evaluate_bezier``."""

    def test_straight_line(self) -> None:
        """A linear Bezier (2 CPs) should produce a straight line."""
        cp = np.array([[0.0, 0.0], [10.0, 5.0]])
        pts = evaluate_bezier(cp, num_samples=11)
        assert pts.shape == (11, 2)
        # Midpoint should be (5, 2.5).
        np.testing.assert_allclose(pts[5], [5.0, 2.5], atol=1e-12)

    def test_start_and_end(self) -> None:
        """Curve must pass through first and last control points."""
        cp = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
        pts = evaluate_bezier(cp, num_samples=101)
        np.testing.assert_allclose(pts[0], cp[0], atol=1e-12)
        np.testing.assert_allclose(pts[-1], cp[-1], atol=1e-12)


class TestFitBezierLsq:
    """Tests for ``fit_bezier_lsq``."""

    def test_round_trip_straight_line(self) -> None:
        """Fitting to points on a line should recover the line."""
        data = np.column_stack([
            np.linspace(0, 10, 50),
            np.linspace(0, 5, 50),
        ])
        cp = fit_bezier_lsq(data, order=3, fix_first=True)
        assert cp.shape == (4, 2)
        # First CP locked to data[0].
        np.testing.assert_allclose(cp[0], data[0], atol=1e-12)
        # Reconstructed curve should be close to input.
        recon = evaluate_bezier(cp, num_samples=50)
        np.testing.assert_allclose(recon, data, atol=0.1)

    def test_fix_first_true(self) -> None:
        """When fix_first=True, first CP must equal first data point."""
        data = np.random.default_rng(42).uniform(0, 10, (30, 2))
        cp = fit_bezier_lsq(data, order=5, fix_first=True)
        np.testing.assert_allclose(cp[0], data[0], atol=1e-12)

    def test_output_shape(self) -> None:
        """Control points must have shape (order+1, 2)."""
        data = np.random.default_rng(0).uniform(0, 10, (40, 2))
        cp = fit_bezier_lsq(data, order=8, fix_first=False)
        assert cp.shape == (9, 2)


class TestCurvatureAt:
    """Tests for ``curvature_at``."""

    def test_straight_line_zero_curvature(self) -> None:
        """A linear Bezier should have zero curvature everywhere."""
        cp = np.array([[0.0, 0.0], [10.0, 0.0]])
        k = curvature_at(cp, t_val=0.5)
        assert abs(k) < 1e-6

    def test_circle_positive_curvature(self) -> None:
        """An approximate semicircle should have positive curvature."""
        # Approximate a quarter-circle with 3 control points.
        r = 5.0
        cp = np.array([
            [r, 0.0],
            [r, r * 0.55228],  # Cubic Bezier circle approx.
            [r * 0.55228, r],
        ])
        k = curvature_at(cp, t_val=0.5)
        # Curvature should be roughly 1/r = 0.2; accept within range.
        assert k > 0


class TestBezierCurve:
    """Tests for the ``BezierCurve`` wrapper class."""

    @pytest.fixture()
    def simple_curve(self) -> BezierCurve:
        """A simple quadratic Bezier for testing."""
        cp = np.array([[0.0, 0.0], [5.0, 10.0], [10.0, 0.0]])
        return BezierCurve(cp)

    def test_evaluate_shape(self, simple_curve: BezierCurve) -> None:
        pts = simple_curve.evaluate(num_samples=50)
        assert pts.shape == (50, 2)

    def test_evaluate_start_at_origin(self, simple_curve: BezierCurve) -> None:
        pts = simple_curve.evaluate()
        np.testing.assert_allclose(pts[0], [0.0, 0.0], atol=1e-12)

    def test_point_at_distance_zero(self, simple_curve: BezierCurve) -> None:
        """Distance 0 should return approximately the start point."""
        pt = simple_curve.point_at_distance(0.0)
        np.testing.assert_allclose(pt, [0.0, 0.0], atol=0.1)

    def test_point_at_distance_positive(
        self, simple_curve: BezierCurve,
    ) -> None:
        """A positive distance should move along the curve."""
        pt = simple_curve.point_at_distance(5.0)
        # Should be somewhere between start and end.
        assert 0.0 < pt[0] < 10.0

    def test_curvature_callable(self, simple_curve: BezierCurve) -> None:
        """Curvature method should return a float."""
        k = simple_curve.curvature(0.5)
        assert isinstance(k, float)

    def test_order_attribute(self, simple_curve: BezierCurve) -> None:
        assert simple_curve.order == 2
