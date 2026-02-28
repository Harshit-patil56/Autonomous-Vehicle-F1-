"""Tests for lap filtering logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_processing.filter_laps import filter_laps


def _make_telemetry(
    laps: list[int],
    current_lap_times: list[float],
    invalid_laps: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal telemetry DataFrame for testing.

    Args:
        laps: Lap number for each row.
        current_lap_times: Current lap time for each row.
        invalid_laps: ``invalid_lap`` flag per row (0/1). Defaults to
            all zeros (all valid).

    Returns:
        DataFrame suitable for ``filter_laps``.
    """
    n = len(laps)
    df = pd.DataFrame({
        "lap": laps,
        "current_lap_time": current_lap_times,
        "game_time": list(range(n)),
    })
    if invalid_laps is not None:
        df["invalid_lap"] = invalid_laps
    else:
        df["invalid_lap"] = 0
    return df


class TestFilterLaps:
    """Tests for ``filter_laps``."""

    def test_removes_lap_zero(self) -> None:
        """Rows with lap==0 (formation/out-lap) must be dropped."""
        df = _make_telemetry(
            laps=[0, 0, 1, 1, 2, 2],
            current_lap_times=[0.0, 1.0, 0.0, 60.0, 0.0, 62.0],
        )
        filtered, report = filter_laps(df, personal_best_seconds=0.0)
        assert (filtered["lap"] > 0).all()

    def test_removes_invalid_laps(self) -> None:
        """Laps flagged as invalid should be fully removed."""
        df = _make_telemetry(
            laps=[1, 1, 2, 2, 3, 3],
            current_lap_times=[0.0, 60.0, 0.0, 61.0, 0.0, 62.0],
            invalid_laps=[0, 0, 1, 1, 0, 0],
        )
        filtered, report = filter_laps(df, personal_best_seconds=0.0)
        assert 2 not in filtered["lap"].values

    def test_removes_slow_laps(self) -> None:
        """Laps > 5% slower than PB must be removed."""
        # PB = 60s, threshold = 63s.
        # Lap 2 has current_lap_time.max() = 70 → dropped.
        df = _make_telemetry(
            laps=[1, 1, 2, 2, 3, 3],
            current_lap_times=[0.0, 60.0, 0.0, 70.0, 0.0, 62.0],
        )
        filtered, report = filter_laps(df, personal_best_seconds=60.0)
        assert 2 not in filtered["lap"].values
        assert 1 in filtered["lap"].values
        assert 3 in filtered["lap"].values

    def test_keeps_fast_laps(self) -> None:
        """Laps within 5% of PB must be kept."""
        # PB = 60s, threshold = 63s.
        df = _make_telemetry(
            laps=[1, 1, 2, 2],
            current_lap_times=[0.0, 60.0, 0.0, 62.0],
        )
        filtered, report = filter_laps(df, personal_best_seconds=60.0)
        assert 1 in filtered["lap"].values
        assert 2 in filtered["lap"].values

    def test_pb_zero_disables_slow_filter(self) -> None:
        """Passing PB=0 should disable the slow-lap filter."""
        df = _make_telemetry(
            laps=[1, 1, 2, 2],
            current_lap_times=[0.0, 60.0, 0.0, 200.0],
        )
        filtered, report = filter_laps(df, personal_best_seconds=0.0)
        assert 2 in filtered["lap"].values

    def test_report_structure(self) -> None:
        """Report dict must contain the expected keys."""
        df = _make_telemetry(
            laps=[1, 1], current_lap_times=[0.0, 60.0],
        )
        _, report = filter_laps(df, personal_best_seconds=60.0)
        expected_keys = {
            "total_rows_before",
            "total_laps_before",
            "rows_after_drop_lap0",
            "invalid_laps_removed",
            "rows_after_drop_invalid",
            "slow_laps_removed",
            "rows_after_drop_slow",
            "total_rows_after",
            "total_laps_after",
        }
        assert expected_keys.issubset(report.keys())

    def test_empty_after_filter(self) -> None:
        """All-invalid input should produce empty output without crash."""
        df = _make_telemetry(
            laps=[1, 1],
            current_lap_times=[0.0, 60.0],
            invalid_laps=[1, 1],
        )
        filtered, _ = filter_laps(df, personal_best_seconds=0.0)
        assert len(filtered) == 0

    def test_index_reset(self) -> None:
        """Filtered output should have a clean RangeIndex."""
        df = _make_telemetry(
            laps=[0, 1, 1],
            current_lap_times=[0.0, 0.0, 60.0],
        )
        filtered, _ = filter_laps(df, personal_best_seconds=0.0)
        assert list(filtered.index) == list(range(len(filtered)))
