"""Merge labeled sessions into train / val Parquet datasets.

Takes one or more ``frames_labeled.parquet`` files (potentially from
different recording sessions), concatenates them, and splits by
**episode_id** so that entire driving sequences are always kept
together.

Every 3rd episode goes to validation; the rest to training.

Usage::

    python -m post_processing.build_dataset \\
        --inputs data/session_A/frames_labeled.parquet \\
                 data/session_B/frames_labeled.parquet \\
        --output-dir datasets

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def build_dataset(
    input_paths: list[str],
    output_dir: str,
    val_every_n: int = 3,
) -> dict:
    """Concatenate labeled parquet files and split by episode.

    Args:
        input_paths: List of paths to ``frames_labeled.parquet`` files.
        output_dir: Directory to write ``train.parquet`` and
            ``val.parquet``.
        val_every_n: Every *n*-th episode is assigned to validation.

    Returns:
        A statistics dict with row and episode counts.

    Raises:
        FileNotFoundError: If any input file does not exist.
    """
    frames: list[pd.DataFrame] = []
    for path in input_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input not found: {path}")
        df = pd.read_parquet(path)
        # Make image_path absolute if it is relative.
        session_dir = os.path.dirname(os.path.abspath(path))
        images_dir = os.path.join(session_dir, "images")
        if "image_path" in df.columns:
            df["image_path"] = df["image_path"].apply(
                lambda p: p if os.path.isabs(p) else os.path.join(images_dir, p)
            )
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # Re-assign globally unique episode ids (original ids overlap across
    # sessions).
    unique_episodes = df_all["episode_id"].unique()
    episode_map = {old: new for new, old in enumerate(sorted(unique_episodes))}
    df_all["episode_id"] = df_all["episode_id"].map(episode_map)

    # Episode-based split: every val_every_n-th episode → val.
    sorted_episodes = sorted(df_all["episode_id"].unique())
    val_episodes = set(sorted_episodes[val_every_n - 1::val_every_n])

    df_train = df_all[~df_all["episode_id"].isin(val_episodes)].copy()
    df_val = df_all[df_all["episode_id"].isin(val_episodes)].copy()

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.parquet")
    val_path = os.path.join(output_dir, "val.parquet")

    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)

    stats = {
        "total_rows": len(df_all),
        "total_episodes": len(sorted_episodes),
        "train_rows": len(df_train),
        "train_episodes": int(df_train["episode_id"].nunique()),
        "val_rows": len(df_val),
        "val_episodes": int(df_val["episode_id"].nunique()),
        "train_path": train_path,
        "val_path": val_path,
    }
    return stats


def main() -> None:
    """Entry point for the build-dataset CLI."""
    parser = argparse.ArgumentParser(
        description="Merge labeled parquet files and create train/val split."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more frames_labeled.parquet paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory for train.parquet and val.parquet (default: datasets).",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=3,
        help="Assign every N-th episode to validation (default: 3).",
    )
    args = parser.parse_args()

    stats = build_dataset(
        input_paths=args.inputs,
        output_dir=args.output_dir,
        val_every_n=args.val_every,
    )

    print("\n--- DATASET BUILD REPORT ---")
    print(f"Total rows:     {stats['total_rows']}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Train rows:     {stats['train_rows']}  "
          f"({stats['train_episodes']} episodes)")
    print(f"Val   rows:     {stats['val_rows']}  "
          f"({stats['val_episodes']} episodes)")
    print(f"\nSaved:")
    print(f"  {stats['train_path']}")
    print(f"  {stats['val_path']}")


if __name__ == "__main__":
    main()
