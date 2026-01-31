import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
MIN_LAP_TIME_SECONDS = 50 
REQUIRED_COLUMNS = ['system_time', 'image_path', 'steer', 'throttle', 'brake', 'speed_kph']
# ---------------------

def get_input_file():
    while True:
        file_path = input(">>> Paste the path to your session folder OR data.csv: ").strip()
        # Remove quotation marks
        file_path = file_path.replace('"', '').replace("'", "")
        
        # 1. If user gives a FOLDER, look for 'data.csv' inside it
        if os.path.isdir(file_path):
            potential_file = os.path.join(file_path, "data.csv")
            if os.path.exists(potential_file):
                return potential_file
            else:
                print(f"[!] Error: I found the folder, but it does not contain 'data.csv'.")
                print(f"    Looked in: {potential_file}\n")
                continue

        # 2. If user gives a FILE, check if it exists
        elif os.path.exists(file_path):
            return file_path
        else:
            print(f"[!] Error: Path not found: '{file_path}'. Try again.\n")

def delete_unused_images(clean_df, images_dir):
    print("\n" + "="*40)
    print(" 🧹 DISK CLEANUP UTILITY")
    print("="*40)
    
    valid_images = set(clean_df['image_path'])
    
    try:
        all_files = os.listdir(images_dir)
    except FileNotFoundError:
        print("Error: Images directory not found.")
        return

    images_to_delete = []
    for filename in all_files:
        if filename.endswith(".jpg") and filename not in valid_images:
            images_to_delete.append(filename)

    count = len(images_to_delete)
    
    if count == 0:
        print("Folder is already clean! No unused images found.")
        return

    print(f"Found {count} unused images (crashes/bad laps).")
    confirm = input(f"⚠️  Do you want to DELETE these {count} files permanently? (y/n): ").lower()
    
    if confirm == 'y':
        print("Deleting files...")
        deleted_count = 0
        for filename in images_to_delete:
            try:
                full_path = os.path.join(images_dir, filename)
                os.remove(full_path)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {filename}: {e}")
        print(f"✅ Deleted {deleted_count} files.")
    else:
        print("Cancelled.")

def clean_telemetry_data():
    print("--- F1 DATASET CLEANER v2.1 ---")
    
    input_file = get_input_file()
    session_dir = os.path.dirname(input_file)
    images_dir = os.path.join(session_dir, "images")
    output_file = os.path.join(session_dir, "cleaned_log.csv")

    print(f"\nLoading {input_file}...")
    try:
        df = pd.read_csv(input_file)
        df = df.sort_values(by='system_time')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Validation
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ CRITICAL ERROR: Missing columns: {missing_cols}")
        return

    if 'lap' not in df.columns:
        print("❌ Error: No 'lap' column found.")
        return

    print("Analyzing laps...")
    df = df[df['lap'] > 0] # Remove grid walk
    
    valid_laps = []
    dropped_laps = []

    for lap_num, lap_data in df.groupby('lap'):
        reasons = []
        
        if 'invalid_lap' in lap_data.columns and lap_data['invalid_lap'].max() == 1:
            reasons.append("Track Limits")

        if 'game_time' in lap_data.columns:
            if (lap_data['game_time'].diff() < 0).any():
                reasons.append("Flashback")

        if 'current_lap_time' in lap_data.columns:
            if lap_data['current_lap_time'].max() < MIN_LAP_TIME_SECONDS:
                 reasons.append("Too Short")

        if not reasons:
            valid_laps.append(lap_data)
        else:
            dropped_laps.append(f"Lap {int(lap_num)}: {', '.join(reasons)}")

    if valid_laps:
        clean_df = pd.concat(valid_laps)
        clean_df.to_csv(output_file, index=False)
        
        print("\n" + "="*40)
        print(f"✅ CSV FILTER COMPLETE")
        print(f"Kept Laps:    {len(valid_laps)}")
        print(f"Dropped Laps: {len(dropped_laps)}")
        print(f"New Index:    {output_file}")
        
        delete_unused_images(clean_df, images_dir)
        
    else:
        print("\n❌ ALL DATA WAS FILTERED OUT! Nothing to save.")

if __name__ == "__main__":
    clean_telemetry_data()