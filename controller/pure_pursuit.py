"""Pure Pursuit steering controller.

Given a smooth path (``BezierCurve``) and the car's current speed,
calculates the steering angle needed to follow the path.

The car is assumed to be at the local-frame origin ``(0, 0)`` heading
in the positive-Z direction.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import math

import numpy as np

from controller.bezier import BezierCurve
from config import CONTROLLER as _CFG


# F1 car approximate wheelbase (meters).
_WHEELBASE: float = _CFG.wheelbase_m

# Maximum physical steering angle (radians).  Normalised output is
# ``steer / _MAX_STEER_RAD`` to produce a [-1, 1] gamepad axis value.
_MAX_STEER_RAD: float = math.radians(_CFG.max_steer_deg)

# Lookahead distance bounds (meters).
_LOOKAHEAD_MIN: float = _CFG.lookahead_min_m
_LOOKAHEAD_MAX: float = _CFG.lookahead_max_m

# Proportional gain mapping speed (m/s) to lookahead (m).
_LOOKAHEAD_GAIN: float = _CFG.lookahead_gain


def compute_lookahead(speed_mps: float) -> float:
    """Compute speed-proportional lookahead distance.

    Args:
        speed_mps: Current car speed in metres per second.

    Returns:
        Lookahead distance in metres, clamped to
        ``[_LOOKAHEAD_MIN, _LOOKAHEAD_MAX]``.
    """
    ld = _LOOKAHEAD_MIN + _LOOKAHEAD_GAIN * speed_mps
    return max(_LOOKAHEAD_MIN, min(ld, _LOOKAHEAD_MAX))


def pure_pursuit_steer(
    curve: BezierCurve,
    speed_kph: float,
) -> float:
    """Compute normalised steering value using Pure Pursuit.

    Algorithm:
        1. Convert speed to m/s and derive lookahead distance ``L``.
        2. Find the point on the curve that is ``L`` metres from origin.
        3. Compute heading angle ``alpha`` to that point.
        4. Apply the Pure Pursuit steering formula.
        5. Normalise to [-1.0, 1.0] for the virtual gamepad.

    Args:
        curve: A ``BezierCurve`` in the car-local frame.
        speed_kph: Current car speed in km/h (from telemetry).

    Returns:
        Normalised steering value in [-1.0, 1.0] where negative is
        left and positive is right.
    """
    speed_mps = speed_kph / 3.6
    lookahead = compute_lookahead(speed_mps)

    # Find the target point on the curve at the lookahead distance.
    target = curve.point_at_distance(lookahead)
    target_x = target[0]  # lateral (car-local X)
    target_z = target[1]  # forward (car-local Z)

    # Angle from car heading (positive Z) to target point.
    alpha = math.atan2(target_x, target_z)

    # Pure Pursuit steering formula:
    #   delta = atan(2 * wheelbase * sin(alpha) / L)
    l_actual = math.sqrt(target_x ** 2 + target_z ** 2)
    if l_actual < 0.01:
        return 0.0

    delta = math.atan2(2.0 * _WHEELBASE * math.sin(alpha), l_actual)

    # Normalise to [-1, 1].
    steer = delta / _MAX_STEER_RAD
    return max(-1.0, min(steer, 1.0))
