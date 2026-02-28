"""Neural network architecture for waypoint prediction.

Input:  single camera frame  (batch, 3, 180, 320) + optional speed
        scalar (batch, 1).
Output: Bezier control points (batch, num_control_points, 2) in the
        car-local coordinate frame (x, z).

The first control point is fixed at (0, 0) — the car's current
position — following the DeepRacing ``fix_first_point`` approach.

Architecture::

    MobileNetV3-Small (pretrained ImageNet, torchvision)
      -> feature vector (batch, 576)
      -> optional concat speed -> (batch, 577)
      -> Linear(577, 256) -> ReLU -> Dropout(0.2)
      -> Linear(256, (num_control_points - 1) * 2)
      -> Reshape -> (batch, num_control_points - 1, 2)
      -> Prepend (0, 0) -> (batch, num_control_points, 2)

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class WaypointNet(nn.Module):
    """Predicts Bezier control points from a single camera frame.

    Attributes:
        num_control_points: Total number of output control points
            (including the fixed origin point).
        use_speed: Whether the speed scalar is concatenated to the
            feature vector before the head.
    """

    def __init__(
        self,
        num_control_points: int = 9,
        use_speed: bool = True,
        dropout: float = 0.2,
    ) -> None:
        """Initialise WaypointNet.

        Args:
            num_control_points: Total Bezier control points (including
                the fixed (0,0) first point).  9 corresponds to an
                order-8 Bezier curve.
            use_speed: If ``True``, expect a speed scalar as a second
                input and concatenate it to the CNN feature vector.
            dropout: Dropout probability in the regression head.
        """
        super().__init__()
        self.num_control_points = num_control_points
        self.use_speed = use_speed

        # --- Backbone ---
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
        )
        # Remove the classification head; keep features + pooling.
        self.features = backbone.features
        self.pool = backbone.avgpool
        # MobileNetV3-Small pool output: (batch, 576, 1, 1) -> flatten
        # to (batch, 576).
        feature_dim: int = 576

        # --- Regression head ---
        head_input_dim = feature_dim + (1 if use_speed else 0)
        # We predict (num_control_points - 1) points because the first
        # point is hardcoded to (0, 0).
        output_dim = (num_control_points - 1) * 2

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        image: torch.Tensor,
        speed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            image: (batch, 3, 180, 320) normalised RGB tensor.
            speed: (batch, 1) normalised speed scalar.  Required when
                ``use_speed=True``.

        Returns:
            (batch, num_control_points, 2) Bezier control points in
            car-local coordinates.  The first point is always (0, 0).
        """
        x = self.features(image)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # (batch, 576)

        if self.use_speed:
            if speed is None:
                raise ValueError(
                    "speed tensor is required when use_speed=True"
                )
            x = torch.cat([x, speed], dim=1)  # (batch, 577)

        out = self.head(x)  # (batch, (cp-1)*2)
        batch_size = out.shape[0]
        predicted = out.view(batch_size, self.num_control_points - 1, 2)

        # Prepend fixed origin point (0, 0).
        origin = torch.zeros(
            batch_size, 1, 2, dtype=out.dtype, device=out.device
        )
        control_points = torch.cat([origin, predicted], dim=1)
        return control_points
