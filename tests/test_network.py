"""Tests for WaypointNet forward pass."""

from __future__ import annotations

import pytest
import torch

from model.network import WaypointNet


class TestWaypointNet:
    """Tests for ``WaypointNet``."""

    @pytest.fixture()
    def net_with_speed(self) -> WaypointNet:
        return WaypointNet(num_control_points=9, use_speed=True, dropout=0.0)

    @pytest.fixture()
    def net_no_speed(self) -> WaypointNet:
        return WaypointNet(num_control_points=9, use_speed=False, dropout=0.0)

    def test_output_shape_with_speed(self, net_with_speed: WaypointNet) -> None:
        batch = 2
        img = torch.randn(batch, 3, 180, 320)
        speed = torch.randn(batch, 1)
        out = net_with_speed(img, speed)
        assert out.shape == (batch, 9, 2)

    def test_output_shape_without_speed(self, net_no_speed: WaypointNet) -> None:
        batch = 2
        img = torch.randn(batch, 3, 180, 320)
        out = net_no_speed(img)
        assert out.shape == (batch, 9, 2)

    def test_first_control_point_is_origin(
        self, net_with_speed: WaypointNet,
    ) -> None:
        """The first control point should always be (0, 0)."""
        img = torch.randn(1, 3, 180, 320)
        speed = torch.randn(1, 1)
        out = net_with_speed(img, speed)
        first_cp = out[0, 0, :]
        torch.testing.assert_close(
            first_cp,
            torch.zeros(2),
            atol=1e-7,
            rtol=0,
        )

    def test_requires_speed_when_use_speed_true(
        self, net_with_speed: WaypointNet,
    ) -> None:
        """Passing speed=None when use_speed=True should raise."""
        img = torch.randn(1, 3, 180, 320)
        with pytest.raises(ValueError, match="speed tensor is required"):
            net_with_speed(img, speed=None)

    def test_no_crash_with_different_batch_sizes(
        self, net_no_speed: WaypointNet,
    ) -> None:
        """Various batch sizes should not crash."""
        for bs in [1, 4]:
            img = torch.randn(bs, 3, 180, 320)
            out = net_no_speed(img)
            assert out.shape[0] == bs

    def test_different_num_control_points(self) -> None:
        net = WaypointNet(num_control_points=5, use_speed=False, dropout=0.0)
        img = torch.randn(1, 3, 180, 320)
        out = net(img)
        assert out.shape == (1, 5, 2)

    def test_gradients_flow(self, net_with_speed: WaypointNet) -> None:
        """Backpropagation should work without errors."""
        img = torch.randn(1, 3, 180, 320)
        speed = torch.randn(1, 1)
        out = net_with_speed(img, speed)
        loss = out.sum()
        loss.backward()
        # At least one parameter should have a gradient.
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in net_with_speed.parameters()
        )
        assert has_grad
