# Autonomous F1 Car v2  
**Synchronized Telemetry & Vision Data Collection for F1 2020**

Autonomous F1 Car v2 is a structured data collection and analysis toolkit designed for research and experimentation in autonomous driving using the **F1 2020 racing simulator**. The system captures high-frequency vehicle telemetry via UDP and synchronizes it with game screenshots to generate datasets suitable for machine learning and performance analysis.

---

## Overview

This project provides:

- Real-time ingestion of F1 2020 UDP telemetry  
- Synchronized visual frame capture aligned with telemetry timestamps  
- Multi-threaded data recording to minimize frame drops  
- Organized, session-based storage of telemetry and images  
- Interactive tools for post-session analysis and visualization  

The output is a clean, structured dataset (CSV + images) that can be directly used for analysis or model training.

---

## Features

### Data Collection

- **Real-time Telemetry Capture**  
  Collects detailed vehicle telemetry from F1 2020 using UDP packets.

- **Synchronized Screen Capture**  
  Captures game screenshots aligned with telemetry frames for vision-based learning.

- **Multi-threaded Architecture**  
  Background workers handle disk I/O and image saving to maintain capture performance.

- **Session Management**  
  Each run is stored in a uniquely timestamped session directory.

---

### Telemetry Data Captured

- **Vehicle Controls**  
  Throttle, brake, steering, gear, DRS status

- **Vehicle Dynamics & Physics**  
  Position (X, Y, Z), yaw, pitch, roll, lateral and longitudinal G-forces

- **Powertrain & Systems**  
  Engine RPM, vehicle speed, engine temperature

- **Track & Lap Information**  
  Lap number, sector, lap distance, lap times, invalid lap flag

- **Tire & Brake Data**  
  Surface and inner tire temperatures, pressures, wheel slip, brake temperatures

---

### Analysis Tools

- **Interactive GUI**  
  Tkinter-based desktop application for telemetry inspection.

- **Time-Series Telemetry Plots**  
  Speed, throttle, brake, and steering traces.

- **Track Mapping**  
  2D visualization of vehicle position and racing lines.

- **Vehicle Dynamics Visualization**  
  G-force plots, tire behavior, and motion characteristics.

- **Lap Analysis**  
  Lap-by-lap comparison of performance metrics.

---

## Requirements

- Windows (recommended for screen capture and window handling)
- Python 3.8 or higher
- F1 2020 with UDP telemetry enabled  
  - Default UDP port: `20777`

---

## Data Storage Format

Each recording session is saved under:

```
data/session_YYYY-MM-DD_HH-MM-SS/
```

### Session Contents

- `data.csv` — synchronized telemetry records  
- `images/` — corresponding screenshots named by timestamp  

### Representative CSV Fields

- `system_time`, `game_time`
- `frame_id`
- `throttle`, `brake`, `steer`
- `gear`, `drs`, `rpm`, `speed_kph`
- `x`, `y`, `z`, `yaw`, `pitch`, `roll`
- `g_lat`, `g_long`, `motion_speed`
- `engine_temp`, `brake_temp`
- `tire_surface`, `tire_inner`, `tire_pressure`, `wheel_slip`
- `lap`, `sector`, `lap_distance`
- `current_lap_time`, `last_lap_time`, `invalid_lap`
- `image_path`

Images are stored as JPEG files at a reduced resolution (default **320×180**) to reduce disk usage and improve downstream processing speed.

---

## Cleaning Session Data

To validate and clean recorded sessions:

```powershell
python data_collection/telementry/clean_data.py
```

The script helps identify incomplete or corrupted telemetry data and removes unused image files.

---

## Project Structure

```
Autonomuns_F1_Car_v2/
├─ data_collection/
│  ├─ telementry/
│  │  ├─ master_recorder.py
│  │  ├─ listener.py
│  │  ├─ state.py
│  │  ├─ packets.py
│  │  ├─ analyze.py
│  │  ├─ clean_data.py
│  │  └─ capture_f1.py
│  ├─ data/
│  └─ model/
├─ Archicture_images/
├─ requirements.txt
├─ Help.md
└─ README.md
```

---

## Configuration

The following parameters can be adjusted in `master_recorder.py`:

- `WINDOW_TITLE_PARTIAL` — substring used to identify the F1 2020 game window  
- `TARGET_SIZE` — screenshot resolution  
- `FLUSH_BATCH_SIZE` — number of telemetry rows buffered before writing to CSV  

Tune these values based on system performance and storage constraints.

---

## Troubleshooting

- **Game window not found**  
  Ensure F1 2020 is running and the window title contains `F1 2020`.

- **Telemetry not received**  
  Verify that UDP telemetry is enabled in-game and port `20777` is open.

- **Permission denied while writing data**  
  Close any application holding the CSV file open or run the script with appropriate permissions.

---

## License & Notice

This project is intended for **research and educational purposes only**.  
Ensure compliance with F1 2020’s terms of service when collecting and using telemetry data.
