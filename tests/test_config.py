"""Tests for the central configuration module."""

from __future__ import annotations

import dataclasses

import pytest

from config import (
    IMAGE,
    MODEL,
    RECORDER,
    INFERENCE,
    CONTROLLER,
    ImageConfig,
    ModelConfig,
    RecorderConfig,
    InferenceConfig,
    ControllerConfig,
)


class TestImageConfig:
    """Verify ImageConfig defaults and immutability."""

    def test_default_height(self) -> None:
        assert IMAGE.height == 180

    def test_default_width(self) -> None:
        assert IMAGE.width == 320

    def test_mean_length(self) -> None:
        assert len(IMAGE.mean) == 3

    def test_std_length(self) -> None:
        assert len(IMAGE.std) == 3

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            IMAGE.height = 999  # type: ignore[misc]


class TestModelConfig:
    """Verify ModelConfig defaults and consistency."""

    def test_bezier_order_equals_cp_minus_one(self) -> None:
        assert MODEL.bezier_order == MODEL.num_control_points - 1

    def test_max_speed_positive(self) -> None:
        assert MODEL.max_speed_kph > 0

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            MODEL.num_control_points = 0  # type: ignore[misc]


class TestRecorderConfig:
    """Verify RecorderConfig defaults."""

    def test_udp_port(self) -> None:
        assert RECORDER.udp_port == 20777

    def test_target_fps_positive(self) -> None:
        assert RECORDER.target_fps > 0


class TestInferenceConfig:
    """Verify InferenceConfig defaults."""

    def test_target_hz_positive(self) -> None:
        assert INFERENCE.target_hz > 0

    def test_onnx_path_ends_with_onnx(self) -> None:
        assert INFERENCE.onnx_path.endswith(".onnx")


class TestControllerConfig:
    """Verify ControllerConfig value ranges."""

    def test_wheelbase_positive(self) -> None:
        assert CONTROLLER.wheelbase_m > 0

    def test_max_steer_positive(self) -> None:
        assert CONTROLLER.max_steer_deg > 0

    def test_lookahead_min_less_than_max(self) -> None:
        assert CONTROLLER.lookahead_min_m < CONTROLLER.lookahead_max_m

    def test_v_max_positive(self) -> None:
        assert CONTROLLER.v_max_kph > 0

    def test_kp_throttle_positive(self) -> None:
        assert CONTROLLER.kp_throttle > 0

    def test_kp_brake_positive(self) -> None:
        assert CONTROLLER.kp_brake > 0

    def test_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            CONTROLLER.wheelbase_m = 0.0  # type: ignore[misc]
