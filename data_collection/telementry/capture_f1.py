import time
import os
import threading
import queue
import cv2
import win32gui
import mss
import numpy as np
import keyboard

# --- CONFIGURATION ---
SEARCH_TERM = "F1 2020" 
OUTPUT_DIR = "training_data/images"
TARGET_SIZE = (320, 180)
TARGET_FPS = 60

class ImageWriterWorker(threading.Thread):
    def __init__(self, output_dir, queue):
        super().__init__()
        self.output_dir = output_dir
        self.queue = queue
        self.running = True
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run(self):
        print("💾 Storage Worker started...")
        while self.running or not self.queue.empty():
            try:
                data = self.queue.get(timeout=0.1)
                timestamp, frame = data
                
                # Resize and Save
                resized_frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                frame_bgr = resized_frame[:, :, :3] # Remove alpha channel
                
                filename = f"{timestamp:.6f}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                cv2.imwrite(filepath, frame_bgr)
                self.queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error saving frame: {e}")

    def stop(self):
        self.running = False

class F1Recorder:
    def __init__(self):
        self.sct = mss.mss()
        self.is_running = False
        self.write_queue = queue.Queue()
        self.writer_thread = ImageWriterWorker(OUTPUT_DIR, self.write_queue)
        self.hwnd = None

    def find_game_window(self):
        print(f"🔍 Searching for window containing: '{SEARCH_TERM}'...")
        found_hwnds = []
        
        def enum_cb(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if SEARCH_TERM.lower() in title.lower():
                    results.append(hwnd)
        
        win32gui.EnumWindows(enum_cb, found_hwnds)
        
        if not found_hwnds:
            raise Exception(f"Game window '{SEARCH_TERM}' not found! Is the game running?")
        
        self.hwnd = found_hwnds[0]
        full_title = win32gui.GetWindowText(self.hwnd)
        rect = win32gui.GetWindowRect(self.hwnd)
        x, y, r, b = rect
        w = r - x
        h = b - y
        
        print(f"✅ Locked onto: '{full_title}'")
        return {"top": y, "left": x, "width": w, "height": h}

    def start(self):
        try:
            region = self.find_game_window()
            self.writer_thread.start()
            self.is_running = True
            
            print(f"🚀 Capture started! Press 'q' to stop.")
            print(f"ℹ️  Note: Recording will PAUSE if '{SEARCH_TERM}' is not in focus.")

            frame_duration = 1.0 / TARGET_FPS

            while self.is_running:
                start_time = time.time()
                
                # --- NEW: AUTO-PAUSE CHECK ---
                active_window = win32gui.GetForegroundWindow()
                
                if active_window == self.hwnd:
                    # Game is active -> Capture
                    img = np.array(self.sct.grab(region))
                    timestamp = time.time()
                    self.write_queue.put((timestamp, img))
                else:
                    # Game is NOT active -> Wait a bit to save CPU
                    time.sleep(0.1) 
                
                # Check for exit key
                if keyboard.is_pressed('q'):
                    print("🛑 Stop signal received...")
                    self.stop()
                    break
                
                # FPS Limiter
                elapsed = time.time() - start_time
                if elapsed < frame_duration:
                    time.sleep(frame_duration - elapsed)
                
        except Exception as e:
            print(f"❌ Error: {e}")
            self.stop()

    def stop(self):
        self.is_running = False
        if self.writer_thread.is_alive():
            print("Finishing write queue...")
            self.writer_thread.stop()
            self.writer_thread.join()
        print("Done.")

if __name__ == "__main__":
    recorder = F1Recorder()
    recorder.start()