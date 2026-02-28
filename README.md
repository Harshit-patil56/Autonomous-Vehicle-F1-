# Autonomous F1 Car v2

**End-to-end autonomous driving for F1 2020 — from data collection to real-time control.**

Camera frames pass through a MobileNetV3-Small CNN that predicts Bezier control points, which are smoothed into a path and followed by a Pure Pursuit steering controller with rule-based throttle/brake and gear logic. The virtual gamepad sends inputs back to the game in real time.

---

## Pipeline

```
Record  →  Filter  →  Label  →  Validate  →  Build  →  Train  →  Export  →  Drive
```

| Stage | Module | Description |
|-------|--------|-------------|
| **Record** | `data_collection/telementry/master_recorder.py` | Captures UDP telemetry + screen at 20 FPS, writes Parquet batches |
| **Filter** | `post_processing/filter_laps.py` | Removes formation laps, invalid laps, slow outliers |
| **Label** | `post_processing/label_waypoints.py` | Generates car-local waypoint labels (vectorized, fast) |
| **Validate** | `post_processing/validate_episodes.py` | Checks temporal integrity, episode structure, data completeness |
| **Build** | `post_processing/build_dataset.py` | Splits data by episode into `train.parquet` / `val.parquet` |
| **Train** | `model/train.py` | PyTorch Lightning training loop (WaypointNet → L1 loss on Bezier CPs) |
| **Export** | `model/export_onnx.py` | Converts `.ckpt` → `.onnx` for fast inference |
| **Drive** | `inference/inference_loop.py` | ONNX inference + controllers → vgamepad output at 50 Hz |

---

## Architecture

```
Camera Frame (320×180 RGB)
        │
  MobileNetV3-Small (pretrained)
        │
  Feature Vector (576) + speed scalar
        │
  Regression Head → 8 Bezier control points
  + fixed origin (0,0) = 9 total
        │
  Bezier Curve (order 8)
        │
  ┌─────┴──────────┐
  Pure Pursuit      Rule-based
  Steering          Throttle/Brake
  │                 │
  └─────┬───────────┘
        │
  Gear Logic (RPM table)
        │
  vgamepad → F1 2020
```

---

## Project Structure

```
Autonomous-Vehicle-F1-/
├── config.py                        # Central frozen-dataclass configuration
├── requirements.txt
├── pytest.ini
├── data_collection/
│   └── telementry/
│       ├── master_recorder.py       # Main recording entry point
│       ├── listener.py              # UDP socket → parsed telemetry
│       ├── state.py                 # Episode tracking, break detection
│       ├── packets.py               # ctypes structs for F1 2020 UDP
│       └── analyze.py               # Tkinter GUI for data inspection
├── post_processing/
│   ├── filter_laps.py               # Lap filtering (invalid, slow, out-lap)
│   ├── label_waypoints.py           # Vectorized waypoint label generation
│   ├── validate_episodes.py         # Data integrity checks
│   └── build_dataset.py             # Train/val split by episode
├── model/
│   ├── network.py                   # WaypointNet (MobileNetV3 + head)
│   ├── dataset.py                   # PyTorch Dataset (Parquet + images)
│   ├── train.py                     # Lightning training CLI
│   └── export_onnx.py               # ONNX export with verification
├── controller/
│   ├── bezier.py                    # Bezier math (fit, evaluate, curvature)
│   ├── pure_pursuit.py              # Steering controller
│   ├── throttle_brake.py            # Speed controller (curvature-based)
│   └── gear_logic.py                # RPM-based gear shifting
├── inference/
│   └── inference_loop.py            # Real-time ONNX + controllers → gamepad
└── tests/
    ├── test_bezier.py
    ├── test_config.py
    ├── test_filter_laps.py
    ├── test_gear_logic.py
    ├── test_label_waypoints.py
    ├── test_network.py
    ├── test_pure_pursuit.py
    └── test_throttle_brake.py
```

---

## Requirements

- **OS:** Windows (required for `dxcam` screen capture and `vgamepad`)
- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA (GTX 1650 4 GB or better)
- **Game:** F1 2020 with UDP telemetry enabled on port `20777`

### Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

### 1. Record data

```powershell
cd data_collection
python -m telementry.master_recorder
```

Telemetry and screenshots are saved to `data_collection/data/session_<timestamp>/` as Parquet batches + JPEG images.

### 2. Filter laps

```powershell
python -m post_processing.filter_laps --session data_collection/data/session_...
```

### 3. Generate waypoint labels

```powershell
python -m post_processing.label_waypoints ^
    --input data_collection/data/session_.../frames_filtered.parquet ^
    --output data_collection/data/session_.../frames_labeled.parquet
```

### 4. Validate episodes

```powershell
python -m post_processing.validate_episodes ^
    --input data_collection/data/session_.../frames_labeled.parquet
```

### 5. Build train/val dataset

```powershell
python -m post_processing.build_dataset ^
    --input data_collection/data/session_.../frames_labeled.parquet ^
    --output-dir data_collection/data/dataset/
```

### 6. Train

```powershell
python -m model.train ^
    --train-parquet data_collection/data/dataset/train.parquet ^
    --val-parquet data_collection/data/dataset/val.parquet ^
    --max-epochs 50 ^
    --seed 42
```

### 7. Export to ONNX

```powershell
python -m model.export_onnx ^
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt ^
    --output model/checkpoints/waypoint_net.onnx
```

### 8. Drive

```powershell
python -m inference.inference_loop
```

The inference loop reads screen frames, runs ONNX inference, and sends steering/throttle/brake/gear commands through a virtual Xbox controller at 50 Hz.

---

## Configuration

All constants are defined in `config.py` using frozen dataclasses:

| Config Class | Controls |
|-------------|----------|
| `ImageConfig` | Input resolution, ImageNet mean/std |
| `ModelConfig` | Control points, Bezier order, speed normalisation |
| `RecorderConfig` | Target FPS, batch size, UDP port |
| `InferenceConfig` | Target Hz, ONNX model path |
| `ControllerConfig` | Wheelbase, max steer, lookahead, throttle/brake gains |

Import the singleton instances directly:

```python
from config import IMAGE, MODEL, CONTROLLER
```

---

## Data Format

Telemetry is stored as **Apache Parquet** (not CSV) for compact storage and fast reads. Key columns:

| Category | Columns |
|----------|---------|
| Position | `x`, `y`, `z`, `yaw`, `pitch`, `roll` |
| Controls | `throttle`, `brake`, `steer`, `gear`, `drs` |
| Dynamics | `speed_kph`, `rpm`, `g_lat`, `g_long` |
| Lap | `lap`, `sector`, `lap_distance`, `current_lap_time`, `invalid_lap` |
| Episode | `episode_id`, `frame_id`, `game_time`, `system_time` |
| Vision | `image_path` |

---

## Episode Tracking

Episodes are continuous, uninterrupted driving sequences. The recorder increments `episode_id` when any of these occur:

- Session or lap restart
- Flashback (time rewind)
- Teleport to pits
- Frame ID or game time decreases
- Position jump > 10 m between frames
- Car stationary (< 1 kph) for > 1 second

Proper episode boundaries prevent **temporal data leakage** in train/val splits — the model never sees future frames during validation.

---

## Testing

```powershell
python -m pytest tests/ -v
```

103 tests covering: Bezier math, Pure Pursuit steering, throttle/brake controller, gear logic, lap filtering, waypoint labelling, neural network forward pass, and configuration integrity.

---

## Analysis GUI

```powershell
cd data_collection
python -m telementry.analyze
```

Tkinter-based tool for inspecting recorded sessions: speed/throttle/brake traces, 2D track maps, tire data, and lap comparisons. Supports both Parquet and CSV files.

---

## License

This project is for **research and educational purposes only**. Comply with F1 2020's terms of service when collecting and using telemetry data.
