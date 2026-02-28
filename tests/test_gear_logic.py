"""Tests for RPM-based gear selection logic."""

from __future__ import annotations

import pytest

from controller.gear_logic import compute_target_gear


class TestComputeTargetGear:
    """Tests for ``compute_target_gear``."""

    def test_no_shift_during_braking(self) -> None:
        """Gear should stay unchanged when braking."""
        gear = compute_target_gear(rpm=12000, current_gear=3, speed_kph=100.0, is_braking=True)
        assert gear == 3

    def test_upshift_at_high_rpm(self) -> None:
        """High RPM in gear 3 should trigger upshift to 4."""
        gear = compute_target_gear(rpm=12000, current_gear=3, speed_kph=200.0, is_braking=False)
        assert gear == 4

    def test_downshift_at_low_rpm(self) -> None:
        """Low RPM in gear 5 should trigger downshift to 4."""
        gear = compute_target_gear(rpm=7000, current_gear=5, speed_kph=100.0, is_braking=False)
        assert gear == 4

    def test_stay_in_gear_normal_rpm(self) -> None:
        """Mid-range RPM should not trigger a shift."""
        gear = compute_target_gear(rpm=10000, current_gear=4, speed_kph=150.0, is_braking=False)
        assert gear == 4

    def test_max_gear_ceiling(self) -> None:
        """Upshift from gear 7 at high RPM should cap at gear 8."""
        gear = compute_target_gear(rpm=12000, current_gear=7, speed_kph=300.0, is_braking=False)
        assert gear == 8

    def test_no_upshift_beyond_max(self) -> None:
        """Gear 8 at high RPM should stay at 8 (no gear 9)."""
        gear = compute_target_gear(rpm=12000, current_gear=8, speed_kph=330.0, is_braking=False)
        assert gear == 8

    def test_min_gear_floor(self) -> None:
        """Downshift from gear 2 at low RPM should floor at gear 1."""
        gear = compute_target_gear(rpm=5000, current_gear=2, speed_kph=30.0, is_braking=False)
        assert gear == 1

    def test_no_downshift_below_min(self) -> None:
        """Gear 1 at low RPM should stay at 1 (no gear 0)."""
        gear = compute_target_gear(rpm=3000, current_gear=1, speed_kph=10.0, is_braking=False)
        assert gear == 1

    def test_neutral_passthrough(self) -> None:
        """Neutral (gear 0) should pass through unchanged."""
        gear = compute_target_gear(rpm=5000, current_gear=0, speed_kph=0.0, is_braking=False)
        assert gear == 0

    def test_reverse_passthrough(self) -> None:
        """Reverse (gear -1) should pass through unchanged."""
        gear = compute_target_gear(rpm=3000, current_gear=-1, speed_kph=5.0, is_braking=False)
        assert gear == -1

    def test_return_type_is_int(self) -> None:
        gear = compute_target_gear(rpm=10000, current_gear=4, speed_kph=150.0)
        assert isinstance(gear, int)

    @pytest.mark.parametrize("gear", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_all_gears_produce_valid_output(self, gear: int) -> None:
        """Every valid gear should produce output in [1, 8]."""
        result = compute_target_gear(rpm=10000, current_gear=gear, speed_kph=150.0)
        assert 1 <= result <= 8
