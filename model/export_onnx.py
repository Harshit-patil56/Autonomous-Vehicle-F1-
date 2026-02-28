"""Export a trained WaypointNet checkpoint to ONNX.

This is the bridge between ``model/train.py`` (produces ``.ckpt``)
and ``inference/inference_loop.py`` (consumes ``.onnx``).

Usage::

    python -m model.export_onnx \
        --checkpoint model/checkpoints/best-epoch=42-val_loss=0.0123.ckpt \
        --output model/checkpoints/waypoint_net.onnx

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from model.train import WaypointLitModule


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    num_control_points: int = 9,
    use_speed: bool = True,
    input_height: int = 180,
    input_width: int = 320,
    opset_version: int = 17,
) -> None:
    """Load a Lightning checkpoint and export to ONNX.

    Args:
        checkpoint_path: Path to ``.ckpt`` file from training.
        output_path: Destination ``.onnx`` file.
        num_control_points: Must match training config.
        use_speed: Must match training config.
        input_height: Image height (must match dataset).
        input_width: Image width (must match dataset).
        opset_version: ONNX opset version.

    Raises:
        FileNotFoundError: If *checkpoint_path* does not exist.
    """
    if not os.path.isfile(checkpoint_path):
        print(f"Error: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Load the Lightning module from checkpoint.
    lit_model = WaypointLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    lit_model.eval()

    # Extract the raw PyTorch model.
    net = lit_model.net
    net.eval()

    # Build dummy inputs matching inference expectations.
    dummy_image = torch.randn(1, 3, input_height, input_width)
    input_names = ["image"]
    dynamic_axes = {"image": {0: "batch"}, "control_points": {0: "batch"}}

    if use_speed:
        dummy_speed = torch.randn(1, 1)
        dummy_inputs = (dummy_image, dummy_speed)
        input_names.append("speed")
        dynamic_axes["speed"] = {0: "batch"}
    else:
        dummy_inputs = (dummy_image,)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        net,
        dummy_inputs,
        output_path,
        input_names=input_names,
        output_names=["control_points"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Verify the exported model loads correctly.
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Exported ONNX model: {output_path}")
    print(f"  Inputs:  {input_names}")
    print(f"  Output:  control_points  (batch, {num_control_points}, 2)")
    print(f"  Size:    {file_size_mb:.2f} MB")
    print(f"  Opset:   {opset_version}")


def main() -> None:
    """Entry point for the ONNX export CLI."""
    parser = argparse.ArgumentParser(
        description="Export a WaypointNet checkpoint to ONNX format.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the Lightning .ckpt file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/checkpoints/waypoint_net.onnx",
        help="Output ONNX file path.",
    )
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Export without speed input.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        use_speed=not args.no_speed,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
