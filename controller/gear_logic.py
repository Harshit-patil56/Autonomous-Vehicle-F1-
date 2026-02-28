"""RPM-based gear selection logic.

Gear shifting is a deterministic function of RPM, current gear, and
whether the car is braking.  There is no reason to train a model for
this — a simple lookup table is optimal.

The shift-point tables below are approximate for a 2020-spec F1
engine.  Tune the values by examining RPM traces from your own
recording sessions in ``analyze.py``.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

# RPM threshold above which the controller requests an upshift.
# Key = current gear, value = RPM threshold for upshift.
_UPSHIFT_RPM: dict[int, int] = {
    1: 11_000,
    2: 11_200,
    3: 11_300,
    4: 11_400,
    5: 11_400,
    6: 11_500,
    7: 11_500,
}

# RPM threshold below which the controller requests a downshift.
# Key = current gear, value = RPM threshold for downshift.
_DOWNSHIFT_RPM: dict[int, int] = {
    2: 7_500,
    3: 8_000,
    4: 8_500,
    5: 9_000,
    6: 9_000,
    7: 9_500,
    8: 9_500,
}

# Minimum and maximum gears the controller will use.
_MIN_GEAR: int = 1
_MAX_GEAR: int = 8


def compute_target_gear(
    rpm: int,
    current_gear: int,
    speed_kph: float,
    is_braking: bool = False,
) -> int:
    """Determine the target gear from RPM and current gear.

    Rules:
        - Never shift during heavy braking (prevents engine-braking
          instability).
        - Upshift when RPM exceeds the per-gear threshold.
        - Downshift when RPM drops below the per-gear threshold.
        - Never exceed ``[_MIN_GEAR, _MAX_GEAR]``.

    Args:
        rpm: Current engine RPM.
        current_gear: Current gear (1-8, or -1 for reverse, 0 for
            neutral).
        speed_kph: Current car speed in km/h.
        is_braking: ``True`` when brake > 0.

    Returns:
        Target gear (int).  Controller should send gear-change commands
        to approach this target.
    """
    # Do not shift while braking.
    if is_braking:
        return current_gear

    # Do not shift in neutral or reverse — let the game handle that.
    if current_gear < _MIN_GEAR:
        return current_gear

    target = current_gear

    # Upshift check.
    if current_gear in _UPSHIFT_RPM:
        if rpm > _UPSHIFT_RPM[current_gear]:
            target = current_gear + 1

    # Downshift check.
    if current_gear in _DOWNSHIFT_RPM:
        if rpm < _DOWNSHIFT_RPM[current_gear]:
            target = current_gear - 1

    # Clamp.
    target = max(_MIN_GEAR, min(target, _MAX_GEAR))

    return target
