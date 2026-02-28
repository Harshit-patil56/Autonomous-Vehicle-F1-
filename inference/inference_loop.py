"""Real-time inference loop for autonomous F1 driving.

Orchestrates:
    1. Screen capture via DXCam (GPU-path, low latency).
    2. ONNX Runtime inference for Bezier control-point prediction.
    3. Pure Pursuit + rule-based throttle/brake/gear controllers.
    4. Virtual gamepad output via vgamepad.

The telemetry listener (``listener.py`` + ``state.py``) runs in a
background thread to provide live speed, RPM, and gear data.

Usage::

    python -m inference.inference_loop --model model/checkpoints/waypoint_net.onnx

Google Python Style Guide: https://google.github.io/styleguide/pyguide.html
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import dxcam
import numpy as np
import onnxruntime as ort
import vgamepad as vg

# ---------------------------------------------------------------------------
# Project-local imports.  Run from the repository root so that
# ``controller/`` and ``data_collection/`` are importable.
# ---------------------------------------------------------------------------
from controller.bezier import BezierCurve
from controller.pure_pursuit import pure_pursuit_steer
from controller.throttle_brake import compute_throttle_brake
from controller.gear_logic import compute_target_gear
from config import IMAGE, MODEL, INFERENCE
from data_collection.telementry.state import TelemetryState
from data_collection.telementry.listener import telemetry_thread

import threading

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Image preprocessing — must match training (model/dataset.py).
_INPUT_HEIGHT: int = IMAGE.height
_INPUT_WIDTH: int = IMAGE.width

# ImageNet normalisation (same as training).
_MEAN = np.array(IMAGE.mean, dtype=np.float32)
_STD = np.array(IMAGE.std, dtype=np.float32)

# Speed normalisation factor (same as dataset.py).
_SPEED_NORM: float = MODEL.max_speed_kph

# Target loop frequency.
_TARGET_HZ: int = INFERENCE.target_hz

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: image preprocessing
# ---------------------------------------------------------------------------

def preprocess_frame(
    frame: np.ndarray,
) -> np.ndarray:
    """Convert a raw BGR capture to a model-ready float32 tensor.

    Steps:
        1. Resize to (_INPUT_HEIGHT, _INPUT_WIDTH).
        2. Convert BGR -> RGB.
        3. Scale to [0, 1].
        4. Normalise with ImageNet mean / std.
        5. Transpose HWC -> CHW and add batch dimension.

    Args:
        frame: Raw BGR uint8 image from DXCam (H, W, 3).

    Returns:
        (1, 3, H, W) float32 numpy array ready for ONNX Runtime.
    """
    resized = cv2.resize(frame, (_INPUT_WIDTH, _INPUT_HEIGHT))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalised = (rgb.astype(np.float32) / 255.0 - _MEAN) / _STD
    chw = np.transpose(normalised, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


# ---------------------------------------------------------------------------
# Helper: reconstruct full control-point array from model output
# ---------------------------------------------------------------------------

def reconstruct_control_points(
    raw_output: np.ndarray,
) -> np.ndarray:
    """Prepend the fixed origin control point to model output.

    The model predicts ``(num_control_points - 1) * 2`` values which
    are reshaped to ``(CP-1, 2)``.  The first control point is always
    ``(0, 0)`` (the car's current position in local frame).

    Args:
        raw_output: (1, (CP-1)*2) raw ONNX output.

    Returns:
        (CP, 2) control points including the origin.
    """
    flat = raw_output[0]  # remove batch dim
    num_pred = len(flat) // 2
    pred_pts = flat.reshape(num_pred, 2)
    origin = np.zeros((1, 2), dtype=np.float32)
    return np.vstack([origin, pred_pts])


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_inference_loop(
    model_path: str,
    use_speed: bool = True,
    providers: Optional[list[str]] = None,
) -> None:
    """Run the real-time autonomous driving loop.

    Args:
        model_path: Path to the exported ONNX model.
        use_speed: Whether the model expects a speed input.
        providers: ONNX Runtime execution providers.  Defaults to
            ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
    """
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # ------------------------------------------------------------------
    # 1. Load ONNX model
    # ------------------------------------------------------------------
    model_file = Path(model_path)
    if not model_file.exists():
        logger.error("Model file not found: %s", model_file)
        sys.exit(1)

    session = ort.InferenceSession(str(model_file), providers=providers)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    logger.info("ONNX model loaded: %s", model_file.name)
    logger.info("  Inputs : %s", input_names)
    logger.info("  Outputs: %s", output_names)

    # ------------------------------------------------------------------
    # 2. Start telemetry listener
    # ------------------------------------------------------------------
    state = TelemetryState()
    telem_thread = threading.Thread(
        target=telemetry_thread,
        args=(state,),
        daemon=True,
    )
    telem_thread.start()
    logger.info("Telemetry listener started (UDP 20777).")

    # ------------------------------------------------------------------
    # 3. Initialise DXCam screen capture
    # ------------------------------------------------------------------
    camera = dxcam.create(output_color="BGR")
    camera.start(target_fps=60, video_mode=True)
    logger.info("DXCam capture started.")

    # ------------------------------------------------------------------
    # 4. Initialise virtual gamepad
    # ------------------------------------------------------------------
    gamepad = vg.VX360Gamepad()
    logger.info("Virtual Xbox 360 gamepad created.")

    # ------------------------------------------------------------------
    # 5. Main control loop
    # ------------------------------------------------------------------
    frame_interval = 1.0 / _TARGET_HZ
    frame_count: int = 0
    fps_counter_start = time.perf_counter()

    logger.info(
        "Entering main loop at %d Hz. Press Ctrl+C to stop.", _TARGET_HZ,
    )

    try:
        while True:
            loop_start = time.perf_counter()

            # --- Capture ---
            frame = camera.get_latest_frame()
            if frame is None:
                time.sleep(0.001)
                continue

            try:
                # --- Preprocess ---
                image_tensor = preprocess_frame(frame)

                # --- Read telemetry ---
                snap = state.snapshot()
                speed_kph: float = float(snap["speed_kph"])
                rpm: int = int(snap["rpm"])
                current_gear: int = int(snap["gear"])

                # --- ONNX inference ---
                feeds: dict[str, np.ndarray] = {"image": image_tensor}
                if use_speed and "speed" in input_names:
                    speed_tensor = np.array(
                        [[speed_kph / _SPEED_NORM]],
                        dtype=np.float32,
                    )
                    feeds["speed"] = speed_tensor

                raw_output = session.run(output_names, feeds)[0]
                control_points = reconstruct_control_points(raw_output)

                # The model already predicts Bezier control points.
                # Use them directly — do NOT re-fit through fit_bezier_lsq,
                # which would treat control points as data points on a
                # curve and distort the prediction.
                curve = BezierCurve(control_points)

                # --- Controllers ---
                steer = pure_pursuit_steer(curve, speed_kph)
                throttle, brake = compute_throttle_brake(curve, speed_kph)
                target_gear = compute_target_gear(
                    rpm, current_gear, speed_kph, is_braking=(brake > 0.05),
                )

            except Exception as frame_err:
                # Fail safe: on any per-frame error, send zero inputs
                # so the car coasts rather than crashing at 300 kph.
                logger.warning("Frame error (safe fallback): %s", frame_err)
                steer = 0.0
                throttle = 0.0
                brake = 1.0
                target_gear = current_gear if "current_gear" in dir() else 1

            # --- Send to virtual gamepad ---
            # Left joystick X: steering [-1, 1] → [-32768, 32767].
            gamepad.left_joystick_float(x_value_float=steer, y_value_float=0.0)
            # Right trigger: throttle [0, 1] → [0, 255].
            gamepad.right_trigger_float(value_float=throttle)
            # Left trigger: brake [0, 1] → [0, 255].
            gamepad.left_trigger_float(value_float=brake)

            # Gear changes — simplified: press A to upshift, X to
            # downshift.  This maps to how F1 2020 handles manual
            # gears with a controller.
            if target_gear > current_gear:
                gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            elif target_gear < current_gear:
                gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
            else:
                gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
                gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)

            gamepad.update()

            # --- FPS logging ---
            frame_count += 1
            elapsed_since_report = time.perf_counter() - fps_counter_start
            if elapsed_since_report >= 2.0:
                fps = frame_count / elapsed_since_report
                logger.info(
                    "FPS: %.1f | speed: %.0f kph | steer: %+.3f | "
                    "throttle: %.2f | brake: %.2f | gear: %d→%d",
                    fps,
                    speed_kph,
                    steer,
                    throttle,
                    brake,
                    current_gear,
                    target_gear,
                )
                frame_count = 0
                fps_counter_start = time.perf_counter()

            # --- Rate limit ---
            elapsed = time.perf_counter() - loop_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received. Shutting down.")
    finally:
        # --- Cleanup ---
        gamepad.reset()
        gamepad.update()
        camera.stop()
        logger.info("Inference loop stopped.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and launch the inference loop."""
    parser = argparse.ArgumentParser(
        description="Run the autonomous F1 driving inference loop.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/checkpoints/waypoint_net.onnx",
        help="Path to the ONNX model file.",
    )
    parser.add_argument(
        "--no-speed",
        action="store_true",
        help="Disable speed input to the model.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution (no CUDA).",
    )
    parser.add_argument(
        "--target-hz",
        type=int,
        default=_TARGET_HZ,
        help="Target loop frequency in Hz.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    providers: list[str]
    if args.cpu:
        providers = ["CPUExecutionProvider"]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Allow overriding the global target Hz.
    global _TARGET_HZ  # noqa: PLW0603
    _TARGET_HZ = args.target_hz

    run_inference_loop(
        model_path=args.model,
        use_speed=not args.no_speed,
        providers=providers,
    )


if __name__ == "__main__":
    main()
