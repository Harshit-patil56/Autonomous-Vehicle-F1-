"""Bezier curve utilities for smoothing predicted waypoints.

Provides:
- ``bezier_matrix``:      Precompute the Bernstein basis matrix.
- ``evaluate_bezier``:    Evaluate a Bezier curve at parameter values.
- ``fit_bezier_lsq``:     Least-squares fit of Bezier control points
                          to a set of data points.
- ``curvature_at``:       Curvature of the Bezier curve at a parameter.
- ``BezierCurve``:        Convenience wrapper around a fitted curve.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

from math import comb

import numpy as np


def bezier_matrix(order: int, t: np.ndarray) -> np.ndarray:
    """Compute the Bernstein polynomial basis matrix.

    Equivalent to DeepRacing's ``bezierM(s, order)``.

    Args:
        order: Bezier order (degree).  For 9 control points, order = 8.
        t: (S,) parameter values in [0, 1].

    Returns:
        (S, order + 1) matrix ``M`` such that
        ``M @ control_points`` evaluates the curve at the given *t*.
    """
    n = order
    k = np.arange(n + 1)
    binom_coeffs = np.array([comb(n, int(ki)) for ki in k], dtype=np.float64)

    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)  # (S, 1)
    # (S, n+1) = C(n,k) * t^k * (1-t)^(n-k)
    basis = binom_coeffs * (t ** k) * ((1.0 - t) ** (n - k))
    return basis


def evaluate_bezier(
    control_points: np.ndarray,
    num_samples: int = 100,
) -> np.ndarray:
    """Evaluate a Bezier curve at ``num_samples`` evenly-spaced points.

    Args:
        control_points: (CP, 2) Bezier control points.
        num_samples: Number of evaluation points.

    Returns:
        (num_samples, 2) curve coordinates.
    """
    order = control_points.shape[0] - 1
    t = np.linspace(0.0, 1.0, num_samples)
    m = bezier_matrix(order, t)
    return m @ control_points


def fit_bezier_lsq(
    points: np.ndarray,
    order: int = 8,
    fix_first: bool = True,
) -> np.ndarray:
    """Least-squares fit of Bezier control points to data points.

    Args:
        points: (N, 2) data points to fit.
        order: Desired Bezier order.
        fix_first: If ``True``, the first control point is forced to
            ``points[0]`` and the fit solves for the remaining.

    Returns:
        (order + 1, 2) fitted control points.
    """
    n_pts = points.shape[0]
    t = np.linspace(0.0, 1.0, n_pts)
    m = bezier_matrix(order, t)  # (N, CP)

    if fix_first:
        # Remove contribution of the first control point and solve
        # for the remaining.
        rhs = points - m[:, 0:1] * points[0:1]
        m_sub = m[:, 1:]
        cp_rest, _, _, _ = np.linalg.lstsq(m_sub, rhs, rcond=None)
        control_points = np.vstack([points[0:1], cp_rest])
    else:
        control_points, _, _, _ = np.linalg.lstsq(m, points, rcond=None)

    return control_points


def curvature_at(
    control_points: np.ndarray,
    t_val: float,
    dt: float = 1e-4,
) -> float:
    """Approximate curvature of the Bezier curve at parameter *t_val*.

    Uses finite-difference approximation of the first and second
    derivatives.

    Args:
        control_points: (CP, 2) Bezier control points.
        t_val: Parameter value in [0, 1].
        dt: Step size for finite differences.

    Returns:
        Signed curvature value.
    """
    order = control_points.shape[0] - 1

    def _eval(t: float) -> np.ndarray:
        t_arr = np.array([np.clip(t, 0.0, 1.0)])
        m = bezier_matrix(order, t_arr)
        return (m @ control_points)[0]

    p0 = _eval(t_val - dt)
    p1 = _eval(t_val)
    p2 = _eval(t_val + dt)

    d1 = (p2 - p0) / (2.0 * dt)
    d2 = (p2 - 2.0 * p1 + p0) / (dt * dt)

    # Curvature: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    denom = (d1[0] ** 2 + d1[1] ** 2) ** 1.5
    if denom < 1e-9:
        return 0.0
    return float((d1[0] * d2[1] - d1[1] * d2[0]) / denom)


class BezierCurve:
    """Convenience wrapper around a set of Bezier control points.

    Stores control points and provides methods to evaluate the curve,
    find a point at a given arc-length distance, and compute curvature.

    Attributes:
        control_points: (CP, 2) numpy array.
        order: Bezier order.
    """

    def __init__(self, control_points: np.ndarray) -> None:
        self.control_points = np.asarray(control_points, dtype=np.float64)
        self.order = self.control_points.shape[0] - 1

    def evaluate(self, num_samples: int = 100) -> np.ndarray:
        """Evaluate the curve at ``num_samples`` evenly-spaced points.

        Returns:
            (num_samples, 2) curve coordinates.
        """
        return evaluate_bezier(self.control_points, num_samples)

    def point_at_distance(
        self,
        target_distance: float,
        num_samples: int = 200,
    ) -> np.ndarray:
        """Find the point on the curve at a given arc-length distance.

        Approximates the arc-length integral by sampling the curve
        densely.

        Args:
            target_distance: Desired arc-length from the start.
            num_samples: Sampling density.

        Returns:
            (2,) point on the curve.
        """
        pts = self.evaluate(num_samples)
        diffs = np.diff(pts, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        total_length = cum_lengths[-1]
        dist = min(target_distance, total_length)

        idx = np.searchsorted(cum_lengths, dist, side="right") - 1
        idx = max(0, min(idx, len(seg_lengths) - 1))

        remaining = dist - cum_lengths[idx]
        if seg_lengths[idx] > 1e-9:
            frac = remaining / seg_lengths[idx]
        else:
            frac = 0.0

        point = pts[idx] + frac * diffs[idx]
        return point

    def curvature(self, t_val: float) -> float:
        """Curvature at parameter *t_val*."""
        return curvature_at(self.control_points, t_val)
