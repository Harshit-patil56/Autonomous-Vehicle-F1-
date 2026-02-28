"""Rule-based throttle and brake controller.

Computes throttle and brake values from the predicted path curvature
and the car's current speed.  Does **not** rely on learned
throttle/brake data because the training data was collected with
ABS ON / TC ON which modifies the raw pedal signals.

Algorithm:
    1. Estimate path curvature at a point slightly ahead.
    2. Derive a target speed: ``v_target = v_max / (1 + kappa * k_gain)``.
    3. Proportional control on the speed error.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

from controller.bezier import BezierCurve
from config import CONTROLLER as _CFG


# Absolute maximum target speed (kph).
_V_MAX_KPH: float = _CFG.v_max_kph

# Gain that maps absolute curvature to speed reduction.  Higher values
# produce more conservative cornering.  Tune on-track.
_CURVATURE_GAIN: float = _CFG.curvature_gain

# Proportional gains for throttle and brake.
_KP_THROTTLE: float = _CFG.kp_throttle
_KP_BRAKE: float = _CFG.kp_brake

# Below this speed (kph) the car is essentially stationary and we
# should apply a standing-start burst of throttle.
_STANDING_START_KPH: float = _CFG.standing_start_kph

# Bezier parameter *t* at which to sample curvature.  0.15 looks a
# short distance ahead of the car.
_CURVATURE_SAMPLE_T: float = _CFG.curvature_sample_t


def compute_throttle_brake(
    curve: BezierCurve,
    speed_kph: float,
) -> tuple[float, float]:
    """Compute throttle and brake from path curvature and speed.

    Args:
        curve: Predicted path in the car-local frame.
        speed_kph: Current car speed in km/h (from telemetry).

    Returns:
        Tuple ``(throttle, brake)`` each in [0.0, 1.0].
    """
    # Standing start — just give full throttle.
    if speed_kph < _STANDING_START_KPH:
        return 1.0, 0.0

    # Curvature at a point slightly ahead.
    kappa = abs(curve.curvature(_CURVATURE_SAMPLE_T))

    # Target speed from curvature.
    v_target = _V_MAX_KPH / (1.0 + kappa * _CURVATURE_GAIN)

    speed_error = v_target - speed_kph

    if speed_error > 0:
        # Too slow — accelerate.
        throttle = min(1.0, speed_error * _KP_THROTTLE)
        brake = 0.0
    else:
        # Too fast — brake.
        throttle = 0.0
        brake = min(1.0, abs(speed_error) * _KP_BRAKE)

    return throttle, brake
