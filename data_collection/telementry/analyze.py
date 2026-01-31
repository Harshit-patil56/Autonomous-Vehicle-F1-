import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import ast

# --- STYLE CONFIGURATION ---
plt.style.use('dark_background')
COLOR_PRIMARY = '#00ffff'    # Cyan
COLOR_THROTTLE = '#00ff00'   # Green
COLOR_BRAKE = '#ff0000'      # Red
BG_COLOR = "#2e2e2e"

class TelemetryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("F1 Telemetry - Constant Distance Fix")
        self.root.geometry("1400x900")
        self.root.configure(bg=BG_COLOR)

        self.df = None
        self.laps_available = []
        self.fastest_lap_num = None

        self._setup_sidebar()
        self._setup_main_area()

    def _setup_sidebar(self):
        control_frame = tk.Frame(self.root, bg=BG_COLOR, width=250, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Label(control_frame, text="Controls", font=("Arial", 16, "bold"), bg=BG_COLOR, fg="white").pack(pady=(0, 20))
        
        tk.Button(control_frame, text="📂 Load CSV", command=self.load_csv, bg="#444", fg="white", font=("Arial", 11)).pack(fill=tk.X, pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=15)
        
        tk.Label(control_frame, text="Select Lap", bg=BG_COLOR, fg="white").pack(anchor="w")
        self.var_lap_primary = tk.StringVar()
        self.combo_lap_primary = ttk.Combobox(control_frame, textvariable=self.var_lap_primary, state="readonly")
        self.combo_lap_primary.pack(fill=tk.X, pady=5)
        self.combo_lap_primary.bind("<<ComboboxSelected>>", self.update_plots)
        
        self.lbl_stats = tk.Label(control_frame, text="Load a CSV to begin.", bg=BG_COLOR, fg="#aaaaaa", justify=tk.LEFT, font=("Consolas", 10))
        self.lbl_stats.pack(fill=tk.X, pady=20, anchor="w")

    def _setup_main_area(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.tab_telemetry = tk.Frame(self.notebook, bg="black")
        self.notebook.add(self.tab_telemetry, text="📈 Telemetry Traces")
        
        self.tab_map = tk.Frame(self.notebook, bg="black")
        self.notebook.add(self.tab_map, text="🗺️ Track Map")

        self.tab_dynamics = tk.Frame(self.notebook, bg="black")
        self.notebook.add(self.tab_dynamics, text="🏎️ Vehicle Dynamics")

        self._init_plots()

    def _init_plots(self):
        self.fig_telem, self.axs_telem = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        self.canvas_telem = FigureCanvasTkAgg(self.fig_telem, master=self.tab_telemetry)
        self.canvas_telem.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas_telem, self.tab_telemetry)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.fig_map, self.ax_map = plt.subplots(figsize=(8, 8))
        self.canvas_map = FigureCanvasTkAgg(self.fig_map, master=self.tab_map)
        self.canvas_map.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.fig_dyn = plt.figure(figsize=(10, 6))
        self.canvas_dyn = FigureCanvasTkAgg(self.fig_dyn, master=self.tab_dynamics)
        self.canvas_dyn.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not path: return
        try:
            raw_df = pd.read_csv(path)
            self.df = self.normalize_data(raw_df)
            self.detect_laps_by_time_reset()
            self.recalculate_distance_per_lap() # The fixed calculation is here
            self.process_laps()
            messagebox.showinfo("Success", f"Loaded {len(self.df)} points.")
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def normalize_data(self, df):
        df.columns = df.columns.str.lower().str.strip()
        rename_map = {
            'speed': 'speed_kph', 'spd': 'speed_kph',
            'throttle_position': 'throttle', 'brake_position': 'brake',
            'steer_angle': 'steer', 'lap_num': 'lap',
            'currenttime': 'current_lap_time', 'time': 'current_lap_time',
            'world_x': 'x', 'world_z': 'z', 'world_y': 'y',
            'g_lat': 'g_lat', 'lateral_g': 'g_lat',
            'g_lon': 'g_long', 'longitudinal_g': 'g_long'
        }
        df = df.rename(columns=rename_map)
        
        # Sort is essential before diff()
        if 'current_lap_time' in df.columns:
            # If lap column exists, sort by Lap then Time
            if 'lap' in df.columns:
                df = df.sort_values(by=['lap', 'current_lap_time'])
            else:
                df = df.sort_values(by=['current_lap_time'])
                
        return df

    def detect_laps_by_time_reset(self):
        if 'current_lap_time' not in self.df.columns: return
        time_vals = self.df['current_lap_time'].values
        lap_ids = np.zeros(len(time_vals), dtype=int)
        current_lap = 1
        for i in range(1, len(time_vals)):
            if time_vals[i] < time_vals[i-1]:
                current_lap += 1
            lap_ids[i] = current_lap
        lap_ids[0] = 1
        self.df['lap'] = lap_ids

    def recalculate_distance_per_lap(self):
        """Forces distance to restart at 0 for every lap and ensures it increases."""
        if 'speed_kph' not in self.df.columns: return

        # 1. Calculate Time Difference (dt)
        # Use a small default (0.016s = 60Hz) if time is missing or identical
        self.df['dt'] = self.df['current_lap_time'].diff().fillna(0.016)
        
        # Fix: Any negative time jump (lap reset) becomes 0
        self.df['dt'] = self.df['dt'].clip(lower=0)
        
        # Fix: If time didn't change (duplicate rows), force a small step 
        # so distance doesn't get stuck at a constant value.
        self.df.loc[self.df['dt'] == 0, 'dt'] = 0.016

        # 2. Calculate Distance Step (Speed in m/s * Time in s)
        speed_ms = self.df['speed_kph'] / 3.6
        self.df['dist_step'] = speed_ms * self.df['dt']
        
        # 3. Cumulative Sum Grouped by Lap
        self.df['lap_distance'] = self.df.groupby('lap')['dist_step'].cumsum()

    def process_laps(self):
        self.laps_available = sorted(self.df['lap'].unique())
        
        # Simple fastest lap logic
        lap_times = self.df.groupby('lap')['current_lap_time'].max()
        valid_laps = lap_times[lap_times > 30] 
        
        if not valid_laps.empty:
            self.fastest_lap_num = valid_laps.idxmin()
        else:
            self.fastest_lap_num = self.laps_available[0] if self.laps_available else 1

        vals = [str(l) for l in self.laps_available]
        self.combo_lap_primary['values'] = vals
        if str(self.fastest_lap_num) in vals:
            self.combo_lap_primary.set(str(self.fastest_lap_num))
        
        self.update_plots()

    def update_plots(self, event=None):
        if self.df is None: return
        lap_num = self.combo_lap_primary.get()
        if not lap_num: return
        
        lap_data = self.df[self.df['lap'] == int(lap_num)]
        if lap_data.empty: return

        # Stats
        t = lap_data['current_lap_time'].max()
        s = lap_data['speed_kph'].max() if 'speed_kph' in lap_data else 0
        self.lbl_stats.config(text=f"Lap {lap_num} | Time: {t:.3f}s | Max Speed: {s:.0f} kph")

        self.draw_telemetry(lap_data)
        self.draw_map(lap_data)
        self.draw_dynamics(lap_data)

    def draw_telemetry(self, data):
        for ax in self.axs_telem:
            ax.clear()
            ax.grid(True, alpha=0.2, linestyle='--')

        x = data['lap_distance']
        
        if 'speed_kph' in data:
            self.axs_telem[0].plot(x, data['speed_kph'], color=COLOR_PRIMARY, linewidth=1.5)
            self.axs_telem[0].set_ylabel("Speed (KPH)")

        if 'throttle' in data:
            self.axs_telem[1].plot(x, data['throttle']*100, color=COLOR_THROTTLE, label='Throttle')
        if 'brake' in data:
            self.axs_telem[1].plot(x, data['brake']*100, color=COLOR_BRAKE, label='Brake')
        self.axs_telem[1].set_ylabel("Input %")
        self.axs_telem[1].legend(loc='upper right', fontsize='small')

        if 'gear' in data:
            self.axs_telem[2].plot(x, data['gear'], color='white')
            self.axs_telem[2].set_ylabel("Gear")

        if 'steer' in data:
            self.axs_telem[3].plot(x, data['steer'], color='yellow')
            self.axs_telem[3].set_ylabel("Steer")
            self.axs_telem[3].set_xlabel("Lap Distance (m)")

        self.fig_telem.tight_layout()
        self.canvas_telem.draw()

    def draw_map(self, data):
        self.ax_map.clear()
        self.ax_map.set_facecolor('black')
        self.ax_map.axis('equal')
        self.ax_map.grid(False)

        x_col = 'x' if 'x' in data else 'world_position_x'
        z_col = 'z' if 'z' in data else 'y' 

        if x_col in data:
            self.ax_map.scatter(data[x_col], data[z_col], c=data['speed_kph'], cmap='plasma', s=4)
            self.ax_map.set_title(f"Track Map")
        else:
            self.ax_map.text(0,0, "No GPS Data", color='white')
        self.canvas_map.draw()

    def draw_dynamics(self, data):
        self.fig_dyn.clf()
        ax_gg = self.fig_dyn.add_subplot(121)
        if 'g_lat' in data and 'g_long' in data:
            ax_gg.scatter(data['g_lat'], data['g_long'], c=data['speed_kph'], cmap='inferno', s=5, alpha=0.6)
            ax_gg.set_title("G-Force")
            ax_gg.grid(True, alpha=0.3)
            ax_gg.axis('equal')

        ax_tires = self.fig_dyn.add_subplot(122)
        tire_cols = ['fl', 'fr', 'rl', 'rr']
        found = False
        for c, color in zip(tire_cols, ['#ff9999', '#99ff99', '#9999ff', '#ffff99']):
            matches = [col for col in data.columns if col.lower() == c]
            if matches:
                ax_tires.plot(data['lap_distance'], data[matches[0]], label=c.upper(), color=color)
                found = True
        
        if not found and 'tire_surface' in data and isinstance(data['tire_surface'].iloc[0], str):
            try:
                tires = pd.DataFrame(data['tire_surface'].apply(ast.literal_eval).to_list(), index=data.index)
                tires.columns = ['FL','FR','RL','RR']
                for c, color in zip(tires.columns, ['#ff9999', '#99ff99', '#9999ff', '#ffff99']):
                     ax_tires.plot(data['lap_distance'], tires[c], label=c, color=color)
                found = True
            except: pass

        if found: ax_tires.legend()
        else: ax_tires.text(0.5, 0.5, "No Tire Data", color="white", ha='center')

        self.fig_dyn.tight_layout()
        self.canvas_dyn.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = TelemetryApp(root)
    root.mainloop()