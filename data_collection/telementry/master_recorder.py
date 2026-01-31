import time
import os
import threading
import queue
import logging
from datetime import datetime
import cv2
import mss
import numpy as np
import pandas as pd

# Import your existing modules
from telementry.state import TelemetryState
from telementry.listener import telemetry_thread

# --- CONFIGURATION ---
WINDOW_TITLE_PARTIAL = "F1 2020"  # To find the game window
TARGET_SIZE = (320, 180)          # Resize for AI model
FLUSH_BATCH_SIZE = 200            # Save to disk every N frames
TARGET_FPS = 20                   # Fixed capture rate (frames per second)

class DataWriterWorker(threading.Thread):
    """
    Background worker that saves Images and CSV rows to disk.
    It separates 'Disk I/O' (slow) from 'Capture' (fast).
    """
    def __init__(self, session_dir, write_queue):
        super().__init__()
        self.session_dir = session_dir
        self.images_dir = os.path.join(session_dir, "images")
        self.csv_path = os.path.join(session_dir, "data.csv")
        self.queue = write_queue
        self.running = True
        self.buffer = [] # Buffer for CSV rows

        # Setup folders
        os.makedirs(self.images_dir, exist_ok=True)

    def run(self):
        print("💾 Storage Worker started...")
        while self.running or not self.queue.empty():
            try:
                # Wait for data (timeout allows checking self.running)
                payload = self.queue.get(timeout=0.1)
                
                # Payload unpacking
                # We expect: (timestamp, raw_image_bytes, telemetry_dict)
                timestamp, raw_mss_img, tele_data = payload

                # --- 1. PROCESS IMAGE ---
                # Convert raw MSS bytes to Numpy
                img_np = np.array(raw_mss_img)
                
                # Resize (Crucial for disk space/training speed)
                # Note: MSS is BGRA, we drop Alpha channel (:3)
                img_small = cv2.resize(img_np[:,:,:3], TARGET_SIZE, interpolation=cv2.INTER_AREA)
                
                # Generate Filename (using the timestamp as the key)
                img_name = f"{timestamp:.6f}.jpg"
                img_path = os.path.join(self.images_dir, img_name)
                
                # Save Image
                cv2.imwrite(img_path, img_small)

                # --- 2. PROCESS CSV ---
                # Add image filename to telemetry data so they are linked
                tele_data['image_path'] = img_name
                self.buffer.append(tele_data)

                # Flush CSV buffer to disk periodically
                if len(self.buffer) >= FLUSH_BATCH_SIZE:
                    self.flush_csv()

                self.queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in Writer: {e}")

        # Final flush on exit
        self.flush_csv()

    def flush_csv(self):
        if not self.buffer:
            return
        
        df = pd.DataFrame(self.buffer)
        
        # Append mode ('a'), write header only if file is new
        file_exists = os.path.isfile(self.csv_path)
        try:
            df.to_csv(self.csv_path, mode='a', header=not file_exists, index=False)
            self.buffer.clear()
        except PermissionError:
            print(f"⚠️  Permission Denied: Could not write CSV to {self.csv_path}")

    def stop(self):
        self.running = False


def find_game_region(title_search):
    """Finds the game window and returns the dictionary for MSS."""
    import win32gui
    
    hwnd = None
    def enum_cb(h, _):
        nonlocal hwnd
        if win32gui.IsWindowVisible(h) and title_search.lower() in win32gui.GetWindowText(h).lower():
            hwnd = h

    win32gui.EnumWindows(enum_cb, None)

    if not hwnd:
        raise Exception(f"Game window '{title_search}' not found!")

    # Get dimensions
    rect = win32gui.GetWindowRect(hwnd)
    x, y, r, b = rect
    w = r - x
    h = b - y
    
    print(f"✅ Locked onto: '{win32gui.GetWindowText(hwnd)}'")
    return {"top": y, "left": x, "width": w, "height": h}, hwnd


def main():
    # --- 1. SETUP SESSION ---
    session_id = f"session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    session_dir = os.path.join("data", session_id)
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n>>> 🏎️  F1 DATA RECORDER v2.0")
    print(f">>> Session: {session_id}")

    # --- 2. SETUP CAMERA & WRITER ---
    try:
        # MSS is fast and thread-safe
        sct = mss.mss()
        # Find the game window area
        region, game_hwnd = find_game_region(WINDOW_TITLE_PARTIAL)
    except Exception as e:
        print(f"❌ Setup Error: {e}")
        return

    # Create the Queue and Worker Thread
    write_queue = queue.Queue()
    writer = DataWriterWorker(session_dir, write_queue)
    writer.start()

    # --- 3. SETUP TELEMETRY ---
    state = TelemetryState()
    udp_thread = threading.Thread(target=telemetry_thread, args=(state,), daemon=True)
    udp_thread.start()
    print(">>> Telemetry Listener Started.")
    
    print(">>> 🟢 RECORDER READY. WAITING FOR GAMEPLAY...")
    print(">>> (Press Ctrl+C to Stop)")

    # --- 4. MAIN RECORDING LOOP ---
    try:
        last_game_time_captured = -1.0
        
        # Episode tracking (for train/val split integrity)
        episode_id = 0
        last_frame_id = -1
        last_lap = -1
        last_position = (0.0, 0.0, 0.0)
        stationary_start_time = None
        STATIONARY_THRESHOLD = 1.0  # Speed < 1 kph
        STATIONARY_DURATION = 1.0   # seconds
        POSITION_JUMP_THRESHOLD = 10.0  # meters
        
        print(f">>> Capturing all telemetry updates (natural rate: ~20 FPS)")
        print(f">>> Using game_time (sessionTime) as primary clock")
        print(f">>> Episode boundary detection: ACTIVE (conservative mode)")
        
        while True:
            # 1. Get the latest data packet
            telemetry_data = state.snapshot()
            current_game_time = telemetry_data['game_time']

            # 2. CHECK: Skip if game hasn't started (game_time == 0)
            if current_game_time == 0:
                time.sleep(0.01)
                continue

            # 3. CHECK: Only capture when game_time advances (new physics state)
            # Skip if same game_time (duplicate) or went backwards (flashback/pause)
            if current_game_time <= last_game_time_captured:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
                continue

            # --- EPISODE BOUNDARY DETECTION (Conservative) ---
            episode_break = False
            
            # Hard Break 1: frame_id goes backward (flashback/reset)
            current_frame_id = telemetry_data['frame_id']
            if last_frame_id > 0 and current_frame_id < last_frame_id:
                episode_break = True
                # print(f"\n[EPISODE] Frame ID decreased: {last_frame_id} -> {current_frame_id}")
            
            # Hard Break 2: game_time goes backward (flashback/pause)
            if last_game_time_captured > 0 and current_game_time < last_game_time_captured:
                episode_break = True
                # print(f"\n[EPISODE] Game time decreased: {last_game_time_captured} -> {current_game_time}")
            
            # Hard Break 3: lap number decreases or resets (lap restart)
            current_lap = telemetry_data['lap']
            if last_lap > 0 and current_lap < last_lap:
                episode_break = True
                # print(f"\n[EPISODE] Lap decreased: {last_lap} -> {current_lap}")
            
            # Soft Break 1: Car stationary for > STATIONARY_DURATION seconds
            current_speed = telemetry_data['speed_kph']
            if current_speed < STATIONARY_THRESHOLD:
                if stationary_start_time is None:
                    stationary_start_time = time.time()
                elif (time.time() - stationary_start_time) > STATIONARY_DURATION:
                    # Only trigger once per stationary period
                    if last_game_time_captured > 0:
                        episode_break = True
                        # print(f"\n[EPISODE] Car stationary for {STATIONARY_DURATION}s")
                    stationary_start_time = None  # Reset to avoid repeated triggers
            else:
                stationary_start_time = None  # Reset when car moves
            
            # Soft Break 2: Large position jump (teleport/crash reset)
            current_position = (telemetry_data['x'], telemetry_data['y'], telemetry_data['z'])
            if last_game_time_captured > 0:
                position_delta = ((current_position[0] - last_position[0])**2 + 
                                 (current_position[1] - last_position[1])**2 + 
                                 (current_position[2] - last_position[2])**2)**0.5
                if position_delta > POSITION_JUMP_THRESHOLD:
                    episode_break = True
                    # print(f"\n[EPISODE] Position jump: {position_delta:.2f}m")
            
            # Increment episode_id if any break detected
            if episode_break:
                episode_id += 1
                print(f"\n[EPISODE] New episode started: episode_id = {episode_id}")

            # 4. 📸 CAPTURE! (Synchronous)
            # We grab the frame NOW - game_time interval is satisfied
            raw_img = sct.grab(region)
            
            # Capture the exact system time for synchronization (not primary clock)
            capture_system_time = time.time()
            
            # Calculate frame_dt for quality validation
            if last_game_time_captured > 0:
                frame_dt = current_game_time - last_game_time_captured
                telemetry_data['frame_dt'] = frame_dt
            else:
                telemetry_data['frame_dt'] = 0.0
            
            # Add episode_id to telemetry data
            telemetry_data['episode_id'] = episode_id
            
            # Update telemetry with timestamps
            telemetry_data['system_time'] = capture_system_time
            # game_time is already in telemetry_data from state.snapshot()
            
            # 5. Send to Writer (Non-blocking)
            write_queue.put((capture_system_time, raw_img, telemetry_data))
            
            # Update tracking variables for next iteration
            last_game_time_captured = current_game_time
            last_frame_id = current_frame_id
            last_lap = current_lap
            last_position = current_position
            
            # 7. Console Status (Every 1 second)
            if int(time.time()) % 2 == 0 and int(time.time() * 10) % 10 == 0:
                 print(f"\r[REC] Lap: {int(telemetry_data['lap'])} | "
                       f"Frame: {telemetry_data['frame_id']} | "
                       f"Queue: {write_queue.qsize()} items", end="")

    except KeyboardInterrupt:
        print("\n\n🔴 Stopping Recorder...")
    
    except Exception as e:
        print(f"\n❌ Loop Error: {e}")

    finally:
        # Cleanup
        writer.stop()
        writer.join()
        print(f">>> Session Saved Successfully to: {session_dir}")

if __name__ == "__main__":
    main()