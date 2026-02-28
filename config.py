"""Central configuration for the Autonomous F1 project.

All magic numbers and shared constants live here so they are defined
exactly once.  Every module imports from ``config`` instead of
hardcoding values.

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ImageConfig:
    """Image preprocessing constants (shared by dataset, inference)."""

    height: int = 180
    width: int = 320
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelConfig:
    """WaypointNet architecture constants."""

    num_control_points: int = 9
    """Total Bezier control points (including fixed origin)."""

    bezier_order: int = 8
    """Bezier degree = num_control_points - 1."""

    num_waypoints: int = 20
    """Future positions used for ground-truth labels."""

    use_speed: bool = True
    """Whether to concatenate speed scalar to CNN features."""

    max_speed_kph: float = 350.0
    """Normalisation constant for the speed scalar."""

    feature_dim: int = 576
    """Output dimension of MobileNetV3-Small global pool."""


@dataclass(frozen=True)
class RecorderConfig:
    """Data collection constants."""

    target_fps: int = 20
    flush_batch_size: int = 200
    udp_port: int = 20777
    window_title: str = "F1 2020"


@dataclass(frozen=True)
class InferenceConfig:
    """Real-time inference loop constants."""

    target_hz: int = 50
    onnx_path: str = "model/checkpoints/waypoint_net.onnx"


@dataclass(frozen=True)
class ControllerConfig:
    """Controller tuning parameters."""

    wheelbase_m: float = 3.6
    max_steer_deg: float = 20.0
    lookahead_min_m: float = 3.0
    lookahead_max_m: float = 12.0
    lookahead_gain: float = 0.15
    v_max_kph: float = 300.0
    curvature_gain: float = 50.0
    kp_throttle: float = 0.02
    kp_brake: float = 0.04
    standing_start_kph: float = 5.0
    curvature_sample_t: float = 0.15


# Singleton instances — import these directly.
IMAGE = ImageConfig()
MODEL = ModelConfig()
RECORDER = RecorderConfig()
INFERENCE = InferenceConfig()
CONTROLLER = ControllerConfig()
