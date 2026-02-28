"""Tests for the vectorized waypoint labelling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_processing.label_waypoints import (
    build_pose_matrices_batch,
    generate_waypoint_labels,
)


class TestBuildPoseMatricesBatch:
    """Tests for ``build_pose_matrices_batch``."""

    def test_output_shape(self) -> None:
        n = 10
        zeros = np.zeros(n)
        mats = build_pose_matrices_batch(zeros, zeros, zeros, zeros, zeros, zeros)
        assert mats.shape == (n, 4, 4)

    def test_identity_at_origin(self) -> None:
        """All angles zero, position at origin → identity matrix."""
        z = np.zeros(1)
        mats = build_pose_matrices_batch(z, z, z, z, z, z)
        np.testing.assert_allclose(mats[0], np.eye(4), atol=1e-14)

    def test_translation_only(self) -> None:
        """Zero angles, non-zero position → rotation block is identity."""
        x = np.array([5.0])
        y = np.array([3.0])
        z_pos = np.array([7.0])
        z_angle = np.zeros(1)
        mats = build_pose_matrices_batch(x, y, z_pos, z_angle, z_angle, z_angle)
        # Translation column.
        np.testing.assert_allclose(mats[0, :3, 3], [5.0, 3.0, 7.0], atol=1e-14)
        # Rotation block should be identity.
        np.testing.assert_allclose(mats[0, :3, :3], np.eye(3), atol=1e-14)

    def test_yaw_90_degrees(self) -> None:
        """90-degree yaw rotates X-axis into Y-axis."""
        yaw = np.array([np.pi / 2])
        z = np.zeros(1)
        mats = build_pose_matrices_batch(z, z, z, yaw, z, z)
        # cos(pi/2) ≈ 0, sin(pi/2) ≈ 1.
        np.testing.assert_allclose(mats[0, 0, 0], 0.0, atol=1e-14)
        np.testing.assert_allclose(mats[0, 1, 0], 1.0, atol=1e-14)

    def test_batch_matches_single(self) -> None:
        """Batch of 3 matrices should match individually built ones."""
        rng = np.random.default_rng(42)
        n = 3
        x = rng.uniform(-100, 100, n)
        y = rng.uniform(-10, 10, n)
        z = rng.uniform(-100, 100, n)
        yaw = rng.uniform(-np.pi, np.pi, n)
        pitch = rng.uniform(-0.3, 0.3, n)
        roll = rng.uniform(-0.3, 0.3, n)

        batch = build_pose_matrices_batch(x, y, z, yaw, pitch, roll)

        for i in range(n):
            single = build_pose_matrices_batch(
                x[i:i + 1], y[i:i + 1], z[i:i + 1],
                yaw[i:i + 1], pitch[i:i + 1], roll[i:i + 1],
            )
            np.testing.assert_allclose(batch[i], single[0], atol=1e-14)

    def test_homogeneous_row(self) -> None:
        """Bottom row of every matrix must be [0, 0, 0, 1]."""
        rng = np.random.default_rng(7)
        n = 5
        ones = rng.uniform(-1, 1, n)
        mats = build_pose_matrices_batch(ones, ones, ones, ones, ones, ones)
        for i in range(n):
            np.testing.assert_allclose(mats[i, 3, :], [0, 0, 0, 1], atol=1e-14)


class TestGenerateWaypointLabels:
    """Tests for ``generate_waypoint_labels``."""

    @pytest.fixture()
    def simple_telemetry(self) -> pd.DataFrame:
        """Create synthetic telemetry: 50 frames, 1 episode, car
        moves forward along the Z axis."""
        n = 50
        return pd.DataFrame({
            "episode_id": [1] * n,
            "x": np.zeros(n),
            "y": np.zeros(n),
            "z": np.linspace(0, 100, n),
            "yaw": np.zeros(n),
            "pitch": np.zeros(n),
            "roll": np.zeros(n),
        })

    def test_output_columns_present(self, simple_telemetry: pd.DataFrame) -> None:
        nw = 5
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=nw)
        for k in range(nw):
            assert f"wp_{k}_x" in result.columns
            assert f"wp_{k}_z" in result.columns

    def test_rows_dropped(self, simple_telemetry: pd.DataFrame) -> None:
        """Last num_waypoints rows of the episode should be dropped."""
        nw = 5
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=nw)
        expected_rows = len(simple_telemetry) - nw
        assert len(result) == expected_rows

    def test_no_nans(self, simple_telemetry: pd.DataFrame) -> None:
        nw = 3
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=nw)
        wp_cols = [f"wp_{k}_{c}" for k in range(nw) for c in ("x", "z")]
        assert result[wp_cols].isna().sum().sum() == 0

    def test_straight_forward_waypoints_x_zero(
        self, simple_telemetry: pd.DataFrame,
    ) -> None:
        """Car drives straight along Z; local X of waypoints should be ~0."""
        nw = 5
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=nw)
        for k in range(nw):
            col = f"wp_{k}_x"
            np.testing.assert_allclose(
                result[col].values, 0.0, atol=1e-10,
                err_msg=f"{col} should be zero for straight-ahead motion",
            )

    def test_straight_forward_waypoints_z_positive(
        self, simple_telemetry: pd.DataFrame,
    ) -> None:
        """Car drives forward; local Z of waypoints should be positive."""
        nw = 5
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=nw)
        for k in range(nw):
            col = f"wp_{k}_z"
            assert (result[col] > 0).all(), f"{col} should all be > 0"

    def test_waypoints_monotonically_further(
        self, simple_telemetry: pd.DataFrame,
    ) -> None:
        """Each successive waypoint should be further ahead in Z."""
        nw = 5
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=nw)
        for _, row in result.iterrows():
            z_vals = [row[f"wp_{k}_z"] for k in range(nw)]
            assert all(
                z_vals[i] < z_vals[i + 1] for i in range(len(z_vals) - 1)
            ), "Z waypoints should be monotonically increasing"

    def test_multi_episode(self) -> None:
        """Waypoints should not cross episode boundaries."""
        n_per = 20
        nw = 5
        ep1 = pd.DataFrame({
            "episode_id": [1] * n_per,
            "x": np.zeros(n_per),
            "y": np.zeros(n_per),
            "z": np.linspace(0, 50, n_per),
            "yaw": np.zeros(n_per),
            "pitch": np.zeros(n_per),
            "roll": np.zeros(n_per),
        })
        ep2 = pd.DataFrame({
            "episode_id": [2] * n_per,
            "x": np.full(n_per, 100.0),  # Different X to distinguish.
            "y": np.zeros(n_per),
            "z": np.linspace(0, 50, n_per),
            "yaw": np.zeros(n_per),
            "pitch": np.zeros(n_per),
            "roll": np.zeros(n_per),
        })
        df = pd.concat([ep1, ep2], ignore_index=True)
        result = generate_waypoint_labels(df, num_waypoints=nw)

        # Each episode should contribute (n_per - nw) rows.
        expected_rows = 2 * (n_per - nw)
        assert len(result) == expected_rows

    def test_missing_columns_raises(self) -> None:
        """Missing required columns should raise ValueError."""
        df = pd.DataFrame({"x": [0.0], "y": [0.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            generate_waypoint_labels(df, num_waypoints=3)

    def test_short_episode_skipped(self) -> None:
        """Episode shorter than num_waypoints should produce 0 rows."""
        nw = 10
        df = pd.DataFrame({
            "episode_id": [1] * 5,
            "x": np.zeros(5),
            "y": np.zeros(5),
            "z": np.arange(5, dtype=float),
            "yaw": np.zeros(5),
            "pitch": np.zeros(5),
            "roll": np.zeros(5),
        })
        result = generate_waypoint_labels(df, num_waypoints=nw)
        assert len(result) == 0

    def test_index_is_clean(self, simple_telemetry: pd.DataFrame) -> None:
        result = generate_waypoint_labels(simple_telemetry, num_waypoints=5)
        assert list(result.index) == list(range(len(result)))
