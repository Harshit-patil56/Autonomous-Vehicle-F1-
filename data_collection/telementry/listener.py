import socket
import ctypes
import time

from telementry.packets import (
    PacketHeader,
    PacketMotionData,
    PacketLapData,
    PacketCarTelemetryData,
    PacketID,
)

# Pre-calculate struct sizes for validation
HEADER_SIZE = ctypes.sizeof(PacketHeader)
MOTION_SIZE = ctypes.sizeof(PacketMotionData)
TELEMETRY_SIZE = ctypes.sizeof(PacketCarTelemetryData)
LAP_DATA_SIZE = ctypes.sizeof(PacketLapData)


def telemetry_thread(shared):
    """
    UDP listener thread that receives F1 2020 telemetry packets.
    
    This thread parses incoming packets and calls shared.update() with
    the extracted data. The update() method is thread-safe.
    
    Args:
        shared: TelemetryState instance from state.py
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 20777))
    print(">>> Telemetry listener started on UDP port 20777")

    while True:
        data, _ = sock.recvfrom(4096)

        # Capture system time immediately on packet arrival
        current_system_time = time.time()

        # Validate minimum packet size
        if len(data) < HEADER_SIZE:
            continue

        # Parse header to determine packet type
        header = PacketHeader.from_buffer_copy(data[:HEADER_SIZE])
        packet_id = header.packetId
        player_idx = header.playerCarIndex
        session_time = header.sessionTime
        frame_id = header.frameIdentifier

        # Common updates for every packet
        updates = {
            'time': session_time,
            'system_time': current_system_time,
            'frame_id': frame_id,
        }

        # ---------------- MOTION DATA (ID 0) ----------------
        if packet_id == PacketID.MOTION:
            if len(data) < MOTION_SIZE:
                continue  # Malformed packet

            packet = PacketMotionData.from_buffer_copy(data)
            motion = packet.carMotionData[player_idx]

            # Calculate speed from velocity vector
            vx = motion.worldVelocityX
            vy = motion.worldVelocityY
            vz = motion.worldVelocityZ
            speed = (vx * vx + vy * vy + vz * vz) ** 0.5

            updates.update({
                'x': motion.worldPositionX,
                'y': motion.worldPositionY,
                'z': motion.worldPositionZ,
                'yaw': motion.yaw,
                'pitch': motion.pitch,
                'roll': motion.roll,
                'g_lat': motion.gForceLateral,
                'g_long': motion.gForceLongitudinal,
                'motion_speed': speed,
                'wheel_slip': list(packet.wheelSlip),
                'wheel_speed': list(packet.wheelSpeed),
            })

        # ---------------- CAR TELEMETRY (ID 6) ----------------
        elif packet_id == PacketID.CAR_TELEMETRY:
            if len(data) < TELEMETRY_SIZE:
                continue  # Malformed packet

            packet = PacketCarTelemetryData.from_buffer_copy(data)
            car = packet.carTelemetryData[player_idx]

            updates.update({
                'throttle': car.throttle,
                'brake': car.brake,
                'steer': car.steer,
                'gear': car.gear,
                'drs': car.drs,
                'speed': car.speed,
                'rpm': car.engineRPM,
                'engine_temp': car.engineTemperature,
                'brake_temp': list(car.brakesTemperature),
                'tire_surface': list(car.tyresSurfaceTemperature),
                'tire_inner': list(car.tyresInnerTemperature),
                'tire_pressure': list(car.tyresPressure),
            })

        # ---------------- LAP DATA (ID 2) ----------------
        elif packet_id == PacketID.LAP_DATA:
            if len(data) < LAP_DATA_SIZE:
                continue  # Malformed packet

            packet = PacketLapData.from_buffer_copy(data)
            lap = packet.lapData[player_idx]

            updates.update({
                'lap': lap.currentLapNum,
                'sector': lap.sector,
                'lap_distance': lap.lapDistance,
                'current_lap_time': lap.currentLapTime,
                'last_lap_time': lap.lastLapTime,
                'invalid_lap': lap.currentLapInvalid,
            })

        # Thread-safe update
        shared.update(updates)