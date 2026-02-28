"""Filter recorded laps to keep only clean, fast driving data.

Reads a session's Parquet telemetry batches and ``session_config.json``,
removes invalid laps (track limits, crashes, slow outliers), and writes
``frames_filtered.parquet`` into the same session directory.

Usage::

    python -m post_processing.filter_laps --session data/session_2026-02-28_12-00-00

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def load_session_frames(session_dir: str) -> pd.DataFrame:
    """Load and concatenate all batch Parquet files from a session.

    Args:
        session_dir: Path to the session directory containing
            ``telemetry_batches/``.

    Returns:
        A single DataFrame with all recorded frames sorted by
        ``game_time``.

    Raises:
        FileNotFoundError: If no batch Parquet files are found.
    """
    batches_dir = os.path.join(session_dir, "telemetry_batches")
    if not os.path.isdir(batches_dir):
        raise FileNotFoundError(
            f"No telemetry_batches directory found in {session_dir}"
        )

    parquet_files = sorted(Path(batches_dir).glob("batch_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No batch_*.parquet files found in {batches_dir}"
        )

    frames: list[pd.DataFrame] = []
    for pf in parquet_files:
        frames.append(pq.read_table(str(pf)).to_pandas())

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("game_time").reset_index(drop=True)
    return df


def load_session_config(session_dir: str) -> dict:
    """Load ``session_config.json`` from a session directory.

    Args:
        session_dir: Path to the session directory.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    config_path = os.path.join(session_dir, "session_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"session_config.json not found in {session_dir}"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_laps(
    df: pd.DataFrame,
    personal_best_seconds: float,
) -> tuple[pd.DataFrame, dict]:
    """Remove invalid and slow laps from the telemetry DataFrame.

    Filtering steps applied sequentially:
    1. Remove rows where ``lap == 0`` (out-lap / formation lap).
    2. Remove rows where ``invalid_lap == 1`` (track-limits violation).
    3. Remove entire laps whose ``last_lap_time`` exceeds
       *personal_best_seconds* × 1.05 (more than 5 % slower than PB).
       This step is skipped when *personal_best_seconds* is ``0.0``.

    Args:
        df: Raw telemetry DataFrame.
        personal_best_seconds: Reference PB lap time.  Set to ``0.0`` to
            disable the PB-based filter.

    Returns:
        A tuple of ``(filtered_df, report)`` where *report* is a dict
        with counts of rows and laps removed at each step.
    """
    report: dict = {
        "total_rows_before": len(df),
        "total_laps_before": int(df["lap"].nunique()),
    }

    # Step 1 — drop formation / out-lap rows.
    df = df[df["lap"] > 0].copy()
    report["rows_after_drop_lap0"] = len(df)

    # Step 2 — drop rows flagged as invalid lap.
    if "invalid_lap" in df.columns:
        invalid_laps = df.loc[df["invalid_lap"] == 1, "lap"].unique()
        df = df[~df["lap"].isin(invalid_laps)]
        report["invalid_laps_removed"] = len(invalid_laps)
    else:
        report["invalid_laps_removed"] = 0
    report["rows_after_drop_invalid"] = len(df)

    # Step 3 — drop entire laps that are > 5 % slower than PB.
    # Use current_lap_time.max() which gives the elapsed time of the
    # lap itself.  The ``last_lap_time`` column in F1 2020 telemetry
    # contains the time of the *previous* lap, not the current one,
    # so it would filter the wrong lap.
    slow_laps_removed: list[int] = []
    if personal_best_seconds > 0:
        threshold = personal_best_seconds * 1.05
        for lap_num, lap_group in df.groupby("lap"):
            lap_time = lap_group["current_lap_time"].max()
            if lap_time > 0 and lap_time > threshold:
                slow_laps_removed.append(int(lap_num))
        if slow_laps_removed:
            df = df[~df["lap"].isin(slow_laps_removed)]
    report["slow_laps_removed"] = slow_laps_removed
    report["rows_after_drop_slow"] = len(df)

    report["total_rows_after"] = len(df)
    report["total_laps_after"] = int(df["lap"].nunique())

    return df.reset_index(drop=True), report


def main() -> None:
    """Entry point for the filter-laps CLI."""
    parser = argparse.ArgumentParser(
        description="Filter recorded laps, keeping only clean fast data."
    )
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Path to the session directory (e.g. data/session_2026-...).",
    )
    args = parser.parse_args()

    session_dir: str = args.session

    # Load data.
    print(f"Loading session from {session_dir} ...")
    df = load_session_frames(session_dir)
    config = load_session_config(session_dir)
    pb = config.get("personal_best_seconds", 0.0)
    print(f"Loaded {len(df)} rows.  PB from config: {pb}s")

    # Filter.
    df_filtered, report = filter_laps(df, personal_best_seconds=pb)

    # Write output.
    out_path = os.path.join(session_dir, "frames_filtered.parquet")
    df_filtered.to_parquet(out_path, index=False)

    # Report.
    print("\n--- FILTER REPORT ---")
    print(f"Rows  before:  {report['total_rows_before']}")
    print(f"Laps  before:  {report['total_laps_before']}")
    print(f"Rows  after dropping lap 0:     {report['rows_after_drop_lap0']}")
    print(f"Invalid laps removed:           {report['invalid_laps_removed']}")
    print(f"Rows  after dropping invalid:   {report['rows_after_drop_invalid']}")
    print(f"Slow laps removed (>{pb * 1.05:.1f}s): {report['slow_laps_removed']}")
    print(f"Rows  after dropping slow:      {report['rows_after_drop_slow']}")
    print(f"Final rows:  {report['total_rows_after']}")
    print(f"Final laps:  {report['total_laps_after']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
