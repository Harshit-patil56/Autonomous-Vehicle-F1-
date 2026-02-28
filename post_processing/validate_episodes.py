"""Validate labeled episode data before training.

Runs quality checks on ``frames_labeled.parquet`` and prints a
human-readable report.  Optionally writes ``episode_report.json``.

Checks performed:
1. Episode statistics (count, frames per episode).
2. Monotonicity of ``game_time`` within each episode.
3. ``frame_dt`` distribution — flags frames with gaps > 0.1 s.
4. Waypoint plausibility — NaN check and distance sanity.

Usage::

    python -m post_processing.validate_episodes \\
        --input data/session_.../frames_labeled.parquet

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def check_episode_stats(df: pd.DataFrame) -> dict:
    """Compute episode-level frame counts.

    Args:
        df: Labeled telemetry DataFrame.

    Returns:
        Dictionary with ``num_episodes``, ``frames_per_episode``
        (min / mean / max), and ``short_episodes`` (< 50 frames).
    """
    ep_counts = df.groupby("episode_id").size()
    short = ep_counts[ep_counts < 50]
    return {
        "num_episodes": int(ep_counts.shape[0]),
        "frames_min": int(ep_counts.min()),
        "frames_mean": round(float(ep_counts.mean()), 1),
        "frames_max": int(ep_counts.max()),
        "short_episodes_lt50": short.index.tolist(),
    }


def check_game_time_monotonicity(df: pd.DataFrame) -> dict:
    """Verify game_time is strictly increasing within each episode.

    Args:
        df: Labeled telemetry DataFrame.

    Returns:
        Dictionary with a list of ``(episode_id, row_index)`` pairs
        where monotonicity is violated.
    """
    violations: list[dict] = []
    for ep_id, ep_group in df.groupby("episode_id"):
        gt = ep_group["game_time"].values
        diffs = np.diff(gt)
        bad_indices = np.where(diffs <= 0)[0]
        for bi in bad_indices:
            global_idx = ep_group.index[int(bi) + 1]
            violations.append({
                "episode_id": int(ep_id),
                "row_index": int(global_idx),
                "prev_game_time": float(gt[bi]),
                "curr_game_time": float(gt[bi + 1]),
            })
    return {"monotonicity_violations": violations}


def check_frame_dt(df: pd.DataFrame, gap_threshold: float = 0.1) -> dict:
    """Flag frames where frame_dt exceeds *gap_threshold*.

    Args:
        df: Labeled telemetry DataFrame (must have ``frame_dt``).
        gap_threshold: Maximum acceptable gap in seconds.

    Returns:
        Dictionary with distribution summary and list of large-gap
        rows.
    """
    if "frame_dt" not in df.columns:
        return {"frame_dt_check": "skipped — column not present"}

    dt = df["frame_dt"].values
    large_gaps = df[df["frame_dt"] > gap_threshold]

    return {
        "frame_dt_min": float(np.nanmin(dt)),
        "frame_dt_mean": round(float(np.nanmean(dt)), 4),
        "frame_dt_max": float(np.nanmax(dt)),
        "frame_dt_std": round(float(np.nanstd(dt)), 4),
        "large_gap_count": int(len(large_gaps)),
        "large_gap_rows": large_gaps.index.tolist()[:20],  # first 20 only
    }


def check_waypoint_labels(
    df: pd.DataFrame,
    num_waypoints: int = 20,
) -> dict:
    """Sanity-check the waypoint label columns.

    - No NaN values in any ``wp_*`` column.
    - The furthest waypoint (``wp_{N-1}``) should be roughly
      5-30 m ahead.  Flag rows outside this range.

    Args:
        df: Labeled telemetry DataFrame.
        num_waypoints: Number of waypoints per frame.

    Returns:
        Dictionary with NaN counts and distance outliers.
    """
    last_x = f"wp_{num_waypoints - 1}_x"
    last_z = f"wp_{num_waypoints - 1}_z"

    wp_cols = [
        f"wp_{k}_{c}" for k in range(num_waypoints) for c in ("x", "z")
    ]

    # Check which expected columns actually exist.
    missing_cols = [c for c in wp_cols if c not in df.columns]
    if missing_cols:
        return {"waypoint_check": f"missing columns: {missing_cols[:5]} ..."}

    nan_count = int(df[wp_cols].isna().sum().sum())

    # Distance of the furthest waypoint from origin (car position).
    dist = np.sqrt(df[last_x].values ** 2 + df[last_z].values ** 2)
    too_close = int(np.sum(dist < 3.0))
    too_far = int(np.sum(dist > 60.0))

    return {
        "nan_count": nan_count,
        "furthest_wp_dist_min": round(float(np.min(dist)), 2),
        "furthest_wp_dist_mean": round(float(np.mean(dist)), 2),
        "furthest_wp_dist_max": round(float(np.max(dist)), 2),
        "rows_wp_too_close_lt3m": too_close,
        "rows_wp_too_far_gt60m": too_far,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def validate(
    df: pd.DataFrame,
    num_waypoints: int = 20,
    gap_threshold: float = 0.1,
) -> dict:
    """Run all validation checks and return a combined report.

    Args:
        df: Labeled telemetry DataFrame.
        num_waypoints: Number of waypoints per frame.
        gap_threshold: Maximum acceptable frame_dt in seconds.

    Returns:
        Combined report dictionary.
    """
    report: dict = {}
    report["episode_stats"] = check_episode_stats(df)
    report["game_time"] = check_game_time_monotonicity(df)
    report["frame_dt"] = check_frame_dt(df, gap_threshold=gap_threshold)
    report["waypoints"] = check_waypoint_labels(df, num_waypoints=num_waypoints)
    return report


def print_report(report: dict) -> None:
    """Pretty-print the validation report to stdout."""
    print("\n=== EPISODE VALIDATION REPORT ===\n")

    ep = report["episode_stats"]
    print(f"Episodes:         {ep['num_episodes']}")
    print(f"Frames per ep:    min={ep['frames_min']}  "
          f"mean={ep['frames_mean']}  max={ep['frames_max']}")
    if ep["short_episodes_lt50"]:
        print(f"  WARNING: {len(ep['short_episodes_lt50'])} episode(s) "
              f"with < 50 frames: {ep['short_episodes_lt50']}")

    gt = report["game_time"]
    n_viol = len(gt["monotonicity_violations"])
    if n_viol == 0:
        print("\nGame-time monotonicity: OK")
    else:
        print(f"\nGame-time monotonicity: FAIL — {n_viol} violation(s)")
        for v in gt["monotonicity_violations"][:5]:
            print(f"  ep {v['episode_id']} row {v['row_index']}: "
                  f"{v['prev_game_time']:.4f} -> {v['curr_game_time']:.4f}")

    dt = report["frame_dt"]
    if isinstance(dt, dict) and "frame_dt_check" in dt:
        print(f"\nFrame-dt check: {dt['frame_dt_check']}")
    else:
        print(f"\nFrame-dt: min={dt['frame_dt_min']:.4f}  "
              f"mean={dt['frame_dt_mean']:.4f}  "
              f"max={dt['frame_dt_max']:.4f}  std={dt['frame_dt_std']:.4f}")
        if dt["large_gap_count"] > 0:
            print(f"  WARNING: {dt['large_gap_count']} frames with dt > 0.1s")

    wp = report["waypoints"]
    if "waypoint_check" in wp:
        print(f"\nWaypoint check: {wp['waypoint_check']}")
    else:
        print(f"\nWaypoint NaNs:  {wp['nan_count']}")
        print(f"Furthest WP dist (m): min={wp['furthest_wp_dist_min']}  "
              f"mean={wp['furthest_wp_dist_mean']}  "
              f"max={wp['furthest_wp_dist_max']}")
        if wp["rows_wp_too_close_lt3m"] > 0:
            print(f"  WARNING: {wp['rows_wp_too_close_lt3m']} rows "
                  f"where furthest WP < 3 m (car nearly stationary?)")
        if wp["rows_wp_too_far_gt60m"] > 0:
            print(f"  WARNING: {wp['rows_wp_too_far_gt60m']} rows "
                  f"where furthest WP > 60 m (possible data error)")


def main() -> None:
    """Entry point for the validate-episodes CLI."""
    parser = argparse.ArgumentParser(
        description="Validate labeled telemetry episodes before training."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to frames_labeled.parquet.",
    )
    parser.add_argument(
        "--num-waypoints",
        type=int,
        default=20,
        help="Number of waypoints per frame (default: 20).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save the report as JSON.",
    )
    args = parser.parse_args()

    input_path: str = args.input
    if not os.path.isfile(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    print(f"Loading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows.")

    report = validate(df, num_waypoints=args.num_waypoints)
    print_report(report)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nReport saved to {args.save_json}")


if __name__ == "__main__":
    main()
