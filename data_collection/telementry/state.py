from dataclasses import dataclass, field
import threading


@dataclass
class ControlState:
    throttle: float = 0.0
    brake: float = 0.0
    steer: float = 0.0
    clutch: float = 0.0
    gear: int = 0
    drs: int = 0
    speed: int = 0
    rpm: int = 0


@dataclass
class VehicleState:
    brake_temp: list = field(default_factory=lambda: [0, 0, 0, 0])
    tire_surface: list = field(default_factory=lambda: [0, 0, 0, 0])
    tire_inner: list = field(default_factory=lambda: [0, 0, 0, 0])
    tire_pressure: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    wheel_speed: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    wheel_slip: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])

    engine_temp: int = 0
    speed: float = 0.0
    g_lat: float = 0.0
    g_long: float = 0.0
    # Position & Orientation
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0


@dataclass
class TrackState:
    lap: int = 0
    sector: int = 0
    lap_distance: float = 0.0
    current_lap_time: float = 0.0
    last_lap_time: float = 0.0
    invalid_lap: int = 0


class TelemetryState:
    """
    Thread-safe container for telemetry data.
    
    The listener thread calls update() to write data.
    The main thread calls snapshot() to read data.
    Both methods use a lock to prevent race conditions.
    """

    def __init__(self):
        self.control = ControlState()
        self.vehicle = VehicleState()
        self.track = TrackState()

        self.time = 0.0           # Game session time
        self.system_time = 0.0    # Real-world time (time.time())
        self.frame_id = 0         # frameIdentifier from UDP header

        self._lock = threading.Lock()

    def update(self, data: dict):
        """
        Thread-safe batch update of telemetry fields.
        Called by the listener thread after parsing a UDP packet.
        """
        with self._lock:
            # --- Timing ---
            if 'time' in data:
                self.time = data['time']
            if 'system_time' in data:
                self.system_time = data['system_time']
            if 'frame_id' in data:
                self.frame_id = data['frame_id']

            # --- Control ---
            if 'throttle' in data:
                self.control.throttle = data['throttle']
            if 'brake' in data:
                self.control.brake = data['brake']
            if 'steer' in data:
                self.control.steer = data['steer']
            if 'clutch' in data:
                self.control.clutch = data['clutch']
            if 'gear' in data:
                self.control.gear = data['gear']
            if 'drs' in data:
                self.control.drs = data['drs']
            if 'speed' in data:
                self.control.speed = data['speed']
            if 'rpm' in data:
                self.control.rpm = data['rpm']

            # --- Vehicle Physics ---
            if 'x' in data:
                self.vehicle.x = data['x']
            if 'y' in data:
                self.vehicle.y = data['y']
            if 'z' in data:
                self.vehicle.z = data['z']
            if 'yaw' in data:
                self.vehicle.yaw = data['yaw']
            if 'pitch' in data:
                self.vehicle.pitch = data['pitch']
            if 'roll' in data:
                self.vehicle.roll = data['roll']
            if 'g_lat' in data:
                self.vehicle.g_lat = data['g_lat']
            if 'g_long' in data:
                self.vehicle.g_long = data['g_long']
            if 'motion_speed' in data:
                self.vehicle.speed = data['motion_speed']
            if 'engine_temp' in data:
                self.vehicle.engine_temp = data['engine_temp']

            # --- Vehicle Arrays ---
            if 'brake_temp' in data:
                self.vehicle.brake_temp = data['brake_temp']
            if 'tire_surface' in data:
                self.vehicle.tire_surface = data['tire_surface']
            if 'tire_inner' in data:
                self.vehicle.tire_inner = data['tire_inner']
            if 'tire_pressure' in data:
                self.vehicle.tire_pressure = data['tire_pressure']
            if 'wheel_speed' in data:
                self.vehicle.wheel_speed = data['wheel_speed']
            if 'wheel_slip' in data:
                self.vehicle.wheel_slip = data['wheel_slip']

            # --- Track ---
            if 'lap' in data:
                self.track.lap = data['lap']
            if 'sector' in data:
                self.track.sector = data['sector']
            if 'lap_distance' in data:
                self.track.lap_distance = data['lap_distance']
            if 'current_lap_time' in data:
                self.track.current_lap_time = data['current_lap_time']
            if 'last_lap_time' in data:
                self.track.last_lap_time = data['last_lap_time']
            if 'invalid_lap' in data:
                self.track.invalid_lap = data['invalid_lap']

    def snapshot(self) -> dict:
        """
        Thread-safe read of all telemetry fields.
        Returns a dictionary suitable for saving to CSV.
        """
        with self._lock:
            return {
                # --- Sync Data ---
                "system_time": self.system_time,
                "game_time": self.time,
                "frame_id": self.frame_id,

                # --- Control ---
                "throttle": self.control.throttle,
                "brake": self.control.brake,
                "steer": self.control.steer,
                "clutch": self.control.clutch,
                "gear": self.control.gear,
                "drs": self.control.drs,
                "rpm": self.control.rpm,
                "speed_kph": self.control.speed,

                # --- Physics ---
                "x": self.vehicle.x,
                "y": self.vehicle.y,
                "z": self.vehicle.z,
                "yaw": self.vehicle.yaw,
                "pitch": self.vehicle.pitch,
                "roll": self.vehicle.roll,
                "g_lat": self.vehicle.g_lat,
                "g_long": self.vehicle.g_long,
                "motion_speed": self.vehicle.speed,
                "engine_temp": self.vehicle.engine_temp,

                # --- Arrays ---
                "brake_temp": list(self.vehicle.brake_temp),
                "tire_surface": list(self.vehicle.tire_surface),
                "tire_inner": list(self.vehicle.tire_inner),
                "tire_pressure": list(self.vehicle.tire_pressure),
                "wheel_slip": list(self.vehicle.wheel_slip),

                # --- Track ---
                "lap": self.track.lap,
                "sector": self.track.sector,
                "lap_distance": self.track.lap_distance,
                "current_lap_time": self.track.current_lap_time,
                "last_lap_time": self.track.last_lap_time,
                "invalid_lap": self.track.invalid_lap,
            }