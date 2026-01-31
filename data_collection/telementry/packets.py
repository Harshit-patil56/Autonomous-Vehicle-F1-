import ctypes

# ---------- Base Types ----------
C_UINT8   = ctypes.c_uint8
C_INT8    = ctypes.c_int8
C_INT16   = ctypes.c_int16
C_UINT16  = ctypes.c_uint16
C_UINT32  = ctypes.c_uint32
C_UINT64  = ctypes.c_uint64
C_FLOAT   = ctypes.c_float

# ---------- Packet IDs ----------
class PacketID:
    MOTION       = 0
    SESSION      = 1
    LAP_DATA     = 2
    EVENT        = 3
    PARTICIPANTS = 4
    SETUPS       = 5
    CAR_TELEMETRY= 6
    CAR_STATUS   = 7

# ---------- Header ----------
class PacketHeader(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("packetFormat", C_UINT16),
        ("gameMajorVersion", C_UINT8),
        ("gameMinorVersion", C_UINT8),
        ("packetVersion", C_UINT8),
        ("packetId", C_UINT8),
        ("sessionUID", C_UINT64),
        ("sessionTime", C_FLOAT),
        ("frameIdentifier", C_UINT32),
        ("playerCarIndex", C_UINT8),
        ("secondaryPlayerCarIndex", C_UINT8),
    ]

# ---------- Packet 0: Motion ----------
class CarMotionData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("worldPositionX", C_FLOAT),
        ("worldPositionY", C_FLOAT),
        ("worldPositionZ", C_FLOAT),
        ("worldVelocityX", C_FLOAT),
        ("worldVelocityY", C_FLOAT),
        ("worldVelocityZ", C_FLOAT),
        ("worldForwardDirX", C_INT16),
        ("worldForwardDirY", C_INT16),
        ("worldForwardDirZ", C_INT16),
        ("worldRightDirX", C_INT16),
        ("worldRightDirY", C_INT16),
        ("worldRightDirZ", C_INT16),
        ("gForceLateral", C_FLOAT),
        ("gForceLongitudinal", C_FLOAT),
        ("gForceVertical", C_FLOAT),
        ("yaw", C_FLOAT),
        ("pitch", C_FLOAT),
        ("roll", C_FLOAT),
    ]

class PacketMotionData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("header", PacketHeader),
        ("carMotionData", CarMotionData * 22),
        ("suspensionPosition", C_FLOAT * 4),
        ("suspensionVelocity", C_FLOAT * 4),
        ("suspensionAcceleration", C_FLOAT * 4),
        ("wheelSpeed", C_FLOAT * 4),
        ("wheelSlip", C_FLOAT * 4),
        ("localVelocityX", C_FLOAT),
        ("localVelocityY", C_FLOAT),
        ("localVelocityZ", C_FLOAT),
        ("angularVelocityX", C_FLOAT),
        ("angularVelocityY", C_FLOAT),
        ("angularVelocityZ", C_FLOAT),
        ("angularAccelerationX", C_FLOAT),
        ("angularAccelerationY", C_FLOAT),
        ("angularAccelerationZ", C_FLOAT),
        ("frontWheelsAngle", C_FLOAT),
    ]

# ---------- Packet 6: Telemetry ----------
class CarTelemetryData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("speed", C_UINT16),
        ("throttle", C_FLOAT),
        ("steer", C_FLOAT),
        ("brake", C_FLOAT),
        ("clutch", C_UINT8),
        ("gear", C_INT8),
        ("engineRPM", C_UINT16),
        ("drs", C_UINT8),
        ("revLightsPercent", C_UINT8),
        ("brakesTemperature", C_UINT16 * 4),
        ("tyresSurfaceTemperature", C_UINT8 * 4),
        ("tyresInnerTemperature", C_UINT8 * 4),
        ("engineTemperature", C_UINT16),
        ("tyresPressure", C_FLOAT * 4),
        ("surfaceType", C_UINT8 * 4),
    ]

class PacketCarTelemetryData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("header", PacketHeader),
        ("carTelemetryData", CarTelemetryData * 22),
        ("buttonStatus", C_UINT32),
        ("mfdPanelIndex", C_UINT8),
        ("mfdPanelIndexSecondaryPlayer", C_UINT8),
        ("suggestedGear", C_INT8),
    ]

# ---------- Packet 2: Lap Data ----------
class LapData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
       ("lastLapTime", ctypes.c_float),       # Float in 2020
        ("currentLapTime", ctypes.c_float),    # Float in 2020
        
        
        ("sector1TimeInMS", ctypes.c_uint16),
        ("sector2TimeInMS", ctypes.c_uint16),
        ("bestLapTime", ctypes.c_float),
        ("bestLapNum", ctypes.c_uint8),
        ("bestLapSector1TimeInMS", ctypes.c_uint16),
        ("bestLapSector2TimeInMS", ctypes.c_uint16),
        ("bestLapSector3TimeInMS", ctypes.c_uint16),
        ("bestOverallSector1TimeInMS", ctypes.c_uint16),
        ("bestOverallSector1LapNum", ctypes.c_uint8),
        ("bestOverallSector2TimeInMS", ctypes.c_uint16),
        ("bestOverallSector2LapNum", ctypes.c_uint8),
        ("bestOverallSector3TimeInMS", ctypes.c_uint16),
        ("bestOverallSector3LapNum", ctypes.c_uint8),
        

        ("lapDistance", ctypes.c_float),
        ("totalDistance", ctypes.c_float),
        ("safetyCarDelta", ctypes.c_float),
        ("carPosition", ctypes.c_uint8),
        ("currentLapNum", ctypes.c_uint8),
        ("pitStatus", ctypes.c_uint8),
        ("sector", ctypes.c_uint8),
        ("currentLapInvalid", ctypes.c_uint8),
        ("penalties", ctypes.c_uint8),
        ("gridPosition", ctypes.c_uint8),
        ("driverStatus", ctypes.c_uint8),
        ("resultStatus", ctypes.c_uint8),
    ]

class PacketLapData(ctypes.LittleEndianStructure):
    _pack_ = 1
    _fields_ = [
        ("header", PacketHeader),
        ("lapData", LapData * 22),
    ]