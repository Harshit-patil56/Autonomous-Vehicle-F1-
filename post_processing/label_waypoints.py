"""Generate waypoint labels from filtered telemetry data.

For every frame *i*, this script looks ahead *N* frames (default 20)
**within the same episode** and transforms each future world position
into the car-local coordinate frame at frame *i*.  The resulting
``(local_x, local_z)`` pairs are the training labels.

Uses the full 4x4 pose-inverse transform (not simplified 2D yaw
rotation) following the DeepRacing approach.  This correctly handles
roll and pitch on banked track sections.

Usage::

    python -m post_processing.label_waypoints \\
        --input data/session_.../frames_filtered.parquet \\
        --output data/session_.../frames_labeled.parquet \\
        --num-waypoints 20

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def build_pose_matrices_batch(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray,
) -> np.ndarray:
    """Construct N 4x4 world-from-car transformation matrices (vectorized).

    The F1 2020 telemetry provides Euler angles in radians.
    Rotation order: ZYX (yaw -> pitch -> roll), the standard vehicle
    convention.

    Args:
        x: (N,) World X positions.
        y: (N,) World Y positions (vertical).
        z: (N,) World Z positions.
        yaw: (N,) Heading angles in radians.
        pitch: (N,) Pitch angles in radians.
        roll: (N,) Roll angles in radians.

    Returns:
        A (N, 4, 4) ``float64`` array of homogeneous transformation
        matrices that map car-local coordinates to world coordinates.
    """
    n = len(x)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # ZYX rotation matrix elements (vectorized).
    mats = np.zeros((n, 4, 4), dtype=np.float64)
    mats[:, 0, 0] = cy * cp
    mats[:, 0, 1] = cy * sp * sr - sy * cr
    mats[:, 0, 2] = cy * sp * cr + sy * sr
    mats[:, 0, 3] = x
    mats[:, 1, 0] = sy * cp
    mats[:, 1, 1] = sy * sp * sr + cy * cr
    mats[:, 1, 2] = sy * sp * cr - cy * sr
    mats[:, 1, 3] = y
    mats[:, 2, 0] = -sp
    mats[:, 2, 1] = cp * sr
    mats[:, 2, 2] = cp * cr
    mats[:, 2, 3] = z
    mats[:, 3, 3] = 1.0
    return mats


def generate_waypoint_labels(
    df: pd.DataFrame,
    num_waypoints: int = 20,
) -> pd.DataFrame:
    """Add waypoint label columns to the telemetry DataFrame.

    For every row *i* the function looks ahead at rows
    ``i+1, i+2, ..., i+num_waypoints`` **within the same
    episode_id**.  Each future position is transformed into the
    car-local frame of row *i*.

    Rows that do not have enough future frames within their episode
    are **dropped** — they cannot form a complete label.

    New columns added: ``wp_0_x, wp_0_z, wp_1_x, wp_1_z, ...``

    The inner computation is fully vectorized using batch matrix
    construction, ``np.linalg.inv`` on (N, 4, 4) arrays, and
    ``np.einsum`` for the coordinate transform — no per-frame Python
    loop.

    Args:
        df: Filtered telemetry DataFrame.  Must contain columns
            ``episode_id, x, y, z, yaw, pitch, roll``.
        num_waypoints: Number of future positions to look ahead.

    Returns:
        A copy of *df* with ``2 * num_waypoints`` new columns and
        incomplete-label rows removed.
    """
    required_cols = {"episode_id", "x", "y", "z", "yaw", "pitch", "roll"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Pre-allocate label arrays (NaN = not computed yet).
    n_rows = len(df)
    labels = np.full((n_rows, num_waypoints, 2), np.nan, dtype=np.float64)

    # Offsets for gathering future positions: [1, 2, ..., num_waypoints].
    offsets = np.arange(1, num_waypoints + 1)

    # Work per-episode so we never cross episode boundaries.
    for _ep_id, ep_group in df.groupby("episode_id"):
        indices = ep_group.index.to_numpy()
        n_ep = len(indices)

        if n_ep <= num_waypoints:
            continue  # Entire episode too short for even one label.

        # Number of valid (labelable) frames in this episode.
        n_valid = n_ep - num_waypoints

        # Extract arrays for the full episode.
        wx = ep_group["x"].values
        wy = ep_group["y"].values
        wz = ep_group["z"].values

        # --- Batch build & invert pose matrices for valid frames ---
        poses = build_pose_matrices_batch(
            wx[:n_valid], wy[:n_valid], wz[:n_valid],
            ep_group["yaw"].values[:n_valid],
            ep_group["pitch"].values[:n_valid],
            ep_group["roll"].values[:n_valid],
        )  # (n_valid, 4, 4)
        poses_inv = np.linalg.inv(poses)  # (n_valid, 4, 4)

        # --- Gather future world positions ---
        # future_idx[i, k] = i + k + 1  (the k-th future frame for frame i).
        future_idx = np.arange(n_valid)[:, np.newaxis] + offsets[np.newaxis, :]
        # Shape: (n_valid, num_waypoints).

        # Build homogeneous future points: (n_valid, num_waypoints, 4).
        future_pts = np.ones(
            (n_valid, num_waypoints, 4), dtype=np.float64,
        )
        future_pts[:, :, 0] = wx[future_idx]
        future_pts[:, :, 1] = wy[future_idx]
        future_pts[:, :, 2] = wz[future_idx]

        # --- Batch transform: pose_inv @ future_pts ---
        # einsum: for each frame n, for each waypoint w:
        #   local[n, w, j] = sum_over_k( poses_inv[n, j, k] * future_pts[n, w, k] )
        local_pts = np.einsum(
            "njk,nwk->nwj", poses_inv, future_pts,
        )  # (n_valid, num_waypoints, 4)

        # Extract local_x (column 0) and local_z (column 2).
        global_indices = indices[:n_valid]
        labels[global_indices, :, 0] = local_pts[:, :, 0]
        labels[global_indices, :, 1] = local_pts[:, :, 2]

    # Build column names and attach to DataFrame.
    col_names: list[str] = []
    for k in range(num_waypoints):
        col_names.append(f"wp_{k}_x")
        col_names.append(f"wp_{k}_z")

    label_flat = labels.reshape(n_rows, num_waypoints * 2)
    label_df = pd.DataFrame(label_flat, columns=col_names, index=df.index)

    df_out = pd.concat([df, label_df], axis=1)

    # Drop rows where any waypoint is NaN (incomplete labels).
    df_out = df_out.dropna(subset=[f"wp_{num_waypoints - 1}_z"]).copy()
    df_out = df_out.reset_index(drop=True)

    return df_out


def main() -> None:
    """Entry point for the label-waypoints CLI."""
    parser = argparse.ArgumentParser(
        description="Generate car-local waypoint labels from filtered telemetry."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to frames_filtered.parquet.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Output path.  Defaults to frames_labeled.parquet in the same "
            "directory as --input."
        ),
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=20,
        help="Number of future waypoints per frame (default: 20).",
    )
    args = parser.parse_args()

    input_path: str = args.input
    num_wp: int = args.num_waypoints

    if not os.path.isfile(input_path):
        print(f"Error: input file not found: {input_path}")
        sys.exit(1)

    output_path: str = args.output
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(input_path), "frames_labeled.parquet"
        )

    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.")

    print(f"Generating {num_wp} waypoint labels per frame ...")
    df_labeled = generate_waypoint_labels(df, num_waypoints=num_wp)
    rows_dropped = len(df) - len(df_labeled)

    df_labeled.to_parquet(output_path, index=False)

    print(f"\n--- LABEL REPORT ---")
    print(f"Input rows:       {len(df)}")
    print(f"Labeled rows:     {len(df_labeled)}")
    print(f"Dropped (tail):   {rows_dropped}")
    print(f"Waypoints/frame:  {num_wp}")
    print(f"New columns:      wp_0_x .. wp_{num_wp - 1}_z  ({num_wp * 2} cols)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
