"""PyTorch Dataset for waypoint-prediction training.

Reads ``train.parquet`` or ``val.parquet`` produced by
``build_dataset.py`` and returns ``(image, waypoints, speed)`` tuples.

Each ``__getitem__`` call:
1. Reads one Parquet row.
2. Loads the linked JPEG image (``image_path``).
3. Optionally applies augmentation (brightness, contrast, h-flip).
4. Normalises with ImageNet mean / std (required for MobileNetV3).
5. Builds the waypoint tensor from ``wp_*`` columns.
6. Returns ``(image_tensor, waypoints_tensor, speed_scalar)``.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import IMAGE as _IMG_CFG, MODEL as _MODEL_CFG

# ImageNet normalisation constants (used by torchvision pretrained
# MobileNetV3).
_IMAGENET_MEAN = np.array(_IMG_CFG.mean, dtype=np.float32)
_IMAGENET_STD = np.array(_IMG_CFG.std, dtype=np.float32)

# Maximum speed used to normalise the speed scalar to roughly [0, 1].
_MAX_SPEED_KPH = _MODEL_CFG.max_speed_kph


class WaypointDataset(Dataset):
    """PyTorch Dataset that yields (image, waypoints, speed) triples.

    Attributes:
        df: Underlying DataFrame loaded from Parquet.
        num_waypoints: Number of waypoints per sample.
        augment: Whether to apply random augmentations.
    """

    def __init__(
        self,
        parquet_path: str,
        num_waypoints: int = 20,
        augment: bool = False,
        target_height: int = 180,
        target_width: int = 320,
    ) -> None:
        """Initialise the dataset.

        Args:
            parquet_path: Path to ``train.parquet`` or ``val.parquet``.
            num_waypoints: Expected number of waypoints per frame.
            augment: Enable random augmentation for training.
            target_height: Image height after resize.
            target_width: Image width after resize.

        Raises:
            FileNotFoundError: If *parquet_path* does not exist.
        """
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Dataset not found: {parquet_path}")

        self.df: pd.DataFrame = pd.read_parquet(parquet_path)
        self.num_waypoints = num_waypoints
        self.augment = augment
        self.target_h = target_height
        self.target_w = target_width

        # Pre-build the list of waypoint column names.
        self._wp_cols: list[str] = []
        for k in range(num_waypoints):
            self._wp_cols.append(f"wp_{k}_x")
            self._wp_cols.append(f"wp_{k}_z")

        # Validate columns exist.
        missing = [c for c in self._wp_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"Missing waypoint columns in parquet: {missing[:4]} ..."
            )
        if "image_path" not in self.df.columns:
            raise ValueError("Missing 'image_path' column in parquet.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a single sample.

        Args:
            idx: Row index into the DataFrame.

        Returns:
            Tuple of:
              - ``image`` — ``(3, H, W)`` float32 tensor.
              - ``waypoints`` — ``(num_waypoints, 2)`` float32 tensor.
              - ``speed`` — ``(1,)`` float32 tensor, normalised by
                ``_MAX_SPEED_KPH``.
        """
        row = self.df.iloc[idx]

        # --- 1. Load image ---
        img_path: str = row["image_path"]
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # BGR -> RGB.
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Resize if needed.
        h, w = img_rgb.shape[:2]
        if h != self.target_h or w != self.target_w:
            img_rgb = cv2.resize(
                img_rgb,
                (self.target_w, self.target_h),
                interpolation=cv2.INTER_AREA,
            )

        # --- 2. Build waypoints ---
        wp_values = row[self._wp_cols].values.astype(np.float32)
        waypoints = wp_values.reshape(self.num_waypoints, 2)

        # --- 3. Augmentation ---
        if self.augment:
            img_rgb, waypoints = self._apply_augmentation(img_rgb, waypoints)

        # --- 4. Normalise image ---
        img_float = img_rgb.astype(np.float32) / 255.0
        img_float = (img_float - _IMAGENET_MEAN) / _IMAGENET_STD

        # HWC -> CHW.
        img_tensor = torch.from_numpy(
            img_float.transpose(2, 0, 1).copy()
        )

        # --- 5. Speed ---
        speed_kph = float(row.get("speed_kph", 0.0))
        speed_norm = np.float32(speed_kph / _MAX_SPEED_KPH)
        speed_tensor = torch.tensor([speed_norm], dtype=torch.float32)

        wp_tensor = torch.from_numpy(waypoints.copy())

        return img_tensor, wp_tensor, speed_tensor

    # -----------------------------------------------------------------
    # Augmentation
    # -----------------------------------------------------------------

    def _apply_augmentation(
        self,
        img: np.ndarray,
        waypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations to the image and waypoints.

        - Brightness shift ±20 %.
        - Contrast shift ±20 %.
        - Horizontal flip (50 % chance) with mirrored waypoints.

        Args:
            img: (H, W, 3) uint8 RGB image.
            waypoints: (N, 2) float32 waypoints.

        Returns:
            Augmented ``(img, waypoints)``.
        """
        rng = np.random.default_rng()

        # Brightness.
        brightness_factor = rng.uniform(0.8, 1.2)
        img = np.clip(img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

        # Contrast.
        contrast_factor = rng.uniform(0.8, 1.2)
        mean_val = img.mean()
        img = np.clip(
            (img.astype(np.float32) - mean_val) * contrast_factor + mean_val,
            0,
            255,
        ).astype(np.uint8)

        # Horizontal flip.
        if rng.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1, :])
            # Mirror local_x (column 0); local_z (column 1) is unchanged.
            waypoints = waypoints.copy()
            waypoints[:, 0] = -waypoints[:, 0]

        return img, waypoints
