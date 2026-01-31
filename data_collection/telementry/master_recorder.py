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
        last_game_time = -1.0
        
        # We limit the loop slightly to prevent 100% CPU usage if telemetry halts
        # But generally, we want to run as fast as telemetry arrives.
        
        while True:
            # 1. Get the latest data packet
            telemetry_data = state.snapshot()
            current_game_time = telemetry_data['game_time']

            # 2. CHECK: Is the game paused or in a flashback?
            # If game_time hasn't changed, or is 0, we skip.
            if current_game_time == last_game_time or current_game_time == 0:
                time.sleep(0.01) # Sleep briefly to yield CPU
                continue

            # 3. CHECK: Is the user actually driving? (Optional safety)
            # You might want to skip if 'lap' == 0 (Grid walk)
            
            # 4. 📸 CAPTURE! (Synchronous)
            # We grab the frame NOW because we know the physics just updated.
            raw_img = sct.grab(region)
            
            # Capture the exact system time of the image
            capture_time = time.time()
            
            # Update the telemetry dictionary with this precise timestamp
            telemetry_data['system_time'] = capture_time
            
            # 5. Send to Writer (Non-blocking)
            write_queue.put((capture_time, raw_img, telemetry_data))
            
            # Update state
            last_game_time = current_game_time
            
            # 6. Console Status (Every 1 second)
            if int(time.time()) % 2 == 0 and int(time.time() * 10) % 10 == 0:
                 print(f"\r[REC] Lap: {int(telemetry_data['lap'])} | "
                       f"Frame: {telemetry_data['frame_id']} | "
                       f"Queue: {write_queue.qsize()} items", end="")
            
            # Small sleep to align roughly with ~60Hz ticks if needed, 
            # but usually reliance on 'game_time' change is enough.
            time.sleep(0.005)

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