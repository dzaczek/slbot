"""
Binary packet encoding/decoding helpers for slither.io WebSocket protocol.

Packet format:
- Server->Client: 2-byte timestamp (uint16) + 1-byte type char + payload
  Exception: pre-init packet '6' has no timestamp prefix
- Client->Server: raw bytes (no header)

Encoding conventions:
- Angles: byte_value * 2*pi / 256
- Speed: byte_value / 18
- Fullness (fam): uint24 / 16777215
- Positions: int16 (absolute) or (byte - 128) / 2 (relative)
"""

import struct
import math

TWO_PI = 2.0 * math.pi


# ─── Reading helpers (from binary buffer) ────────────────────────────────

class PacketReader:
    """Sequential binary reader for slither.io packets."""

    __slots__ = ('data', 'offset')

    def __init__(self, data: bytes, offset: int = 0):
        self.data = data
        self.offset = offset

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def read_uint8(self) -> int:
        val = self.data[self.offset]
        self.offset += 1
        return val

    def read_int8(self) -> int:
        val = struct.unpack_from('b', self.data, self.offset)[0]
        self.offset += 1
        return val

    def read_uint16(self) -> int:
        val = struct.unpack_from('>H', self.data, self.offset)[0]
        self.offset += 2
        return val

    def read_int16(self) -> int:
        val = struct.unpack_from('>h', self.data, self.offset)[0]
        self.offset += 2
        return val

    def read_uint24(self) -> int:
        b = self.data[self.offset:self.offset + 3]
        self.offset += 3
        return (b[0] << 16) | (b[1] << 8) | b[2]

    def read_int24(self) -> int:
        val = self.read_uint24()
        if val & 0x800000:
            val -= 0x1000000
        return val

    def read_uint32(self) -> int:
        val = struct.unpack_from('>I', self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_string(self, length: int = None) -> str:
        """Read a null-terminated or fixed-length string."""
        if length is not None:
            s = self.data[self.offset:self.offset + length]
            self.offset += length
            return s.decode('utf-8', errors='replace').rstrip('\x00')
        # Null-terminated
        end = self.data.index(0, self.offset) if 0 in self.data[self.offset:] else len(self.data)
        s = self.data[self.offset:end].decode('utf-8', errors='replace')
        self.offset = end + 1
        return s

    def read_bytes(self, n: int) -> bytes:
        val = self.data[self.offset:self.offset + n]
        self.offset += n
        return val

    def skip(self, n: int):
        self.offset += n


# ─── Decoding helpers ────────────────────────────────────────────────────

def decode_angle(byte_val: int) -> float:
    """Decode angle from uint8 to radians [0, 2*pi)."""
    return byte_val * TWO_PI / 256.0


def decode_angle_16(val: int) -> float:
    """Decode angle from uint16 to radians [0, 2*pi) — used in some rotation packets."""
    return val * TWO_PI / 65536.0


def decode_speed(byte_val: int) -> float:
    """Decode speed from uint8."""
    return byte_val / 18.0


def decode_fam(uint24_val: int) -> float:
    """Decode fullness (fam) from uint24 to [0, 1]."""
    return uint24_val / 16777215.0


def decode_relative_pos(byte_val: int) -> float:
    """Decode relative position offset from uint8."""
    return (byte_val - 128) / 2.0


# ─── Encoding helpers (Client -> Server) ─────────────────────────────────

def encode_angle(angle_rad: float) -> bytes:
    """
    Encode angle (radians) for sending to server (protocol_version >= 5).

    The mouse-based angle is a single byte: floor(251 * ang / 2pi), range [0, 250].
    Bytes 251=ping, 252=keyboard, 253=boost_start, 254=boost_stop are reserved.
    """
    angle_rad = angle_rad % TWO_PI
    byte_val = int(251.0 * angle_rad / TWO_PI) & 0xFF
    return bytes([byte_val])


def encode_angle_precise(angle_rad: float) -> bytes:
    """Encode angle with uint16 precision (packet 252)."""
    angle_rad = angle_rad % TWO_PI
    val = int(angle_rad * 65536.0 / TWO_PI) & 0xFFFF
    return struct.pack('>BH', 252, val)


def encode_setup_request() -> bytes:
    """Encode the initial setup request byte [1] — tells server to include timestamps."""
    return bytes([1])


def encode_start_login() -> bytes:
    """Encode the 'c' packet (cstr) that starts the login handshake. Includes null terminator."""
    return bytes([99, 0])  # 'c' + null


# Client version and password from game JS (game1107241958.js)
CLIENT_VERSION = 291
CLIENT_PASSWORD = bytes([
    54, 206, 204, 169, 97, 178, 74, 136, 124, 117,
    14, 210, 106, 236, 8, 208, 136, 213, 140, 111
])


def encode_login(nickname: str, skin_id: int = 0) -> bytes:
    """
    Encode login packet (type 115 = 's').

    Format (from game JS ws.onopen):
      [115] [30] [client_version>>8] [client_version&0xFF]
      [cpw × 20 bytes] [skin_id] [nick_length] [nick_bytes...] [0] [0xFF]
    """
    nick_bytes = nickname.encode('utf-8')[:24]  # Max 24 bytes
    buf = bytearray()
    buf.append(115)                                 # Packet type 's' = login
    buf.append(30)                                  # Fixed byte (from game JS)
    buf.append((CLIENT_VERSION >> 8) & 0xFF)        # client_version high byte (1)
    buf.append(CLIENT_VERSION & 0xFF)               # client_version low byte (35)
    buf.extend(CLIENT_PASSWORD)                     # 20-byte password
    buf.append(skin_id & 0xFF)                      # Skin ID
    buf.append(len(nick_bytes))                     # Nickname length
    buf.extend(nick_bytes)                          # Nickname
    buf.append(0)                                   # Null terminator
    buf.append(0xFF)                                # End marker
    return bytes(buf)


def encode_boost_start() -> bytes:
    """Encode boost start command (packet 253)."""
    return bytes([253])


def encode_boost_stop() -> bytes:
    """Encode boost stop command (packet 254)."""
    return bytes([254])


def encode_ping() -> bytes:
    """Encode ping/keepalive (packet 251 with no data — some servers accept empty)."""
    # The standard keepalive is sending packet 251 periodically
    # But the actual ping is typically handled by the WebSocket layer
    return bytes([251])


# ─── Packet type identification ──────────────────────────────────────────

# Server -> Client packet types (the type byte after 2-byte timestamp)
PACKET_INIT = ord('a')           # Init config
PACKET_SNAKE_ADD = ord('s')      # Add/remove snake
PACKET_MOVE_ABS = ord('g')       # Movement absolute
PACKET_MOVE_ABS2 = ord('G')     # Movement absolute (variant)
PACKET_MOVE_GROW = ord('n')      # Movement + grow
PACKET_MOVE_GROW2 = ord('N')    # Movement + grow (variant)
PACKET_ROTATE_E = ord('e')      # Rotation
PACKET_ROTATE_E2 = ord('E')     # Rotation (variant)
PACKET_ROTATE_3 = ord('3')      # Rotation (ang + speed)
PACKET_ROTATE_4 = ord('4')      # Rotation (variant)
PACKET_ROTATE_5 = ord('5')      # Rotation (variant)
PACKET_FOOD_ADD = ord('F')      # Add food (natural spawn)
PACKET_FOOD_ADD_B = ord('b')    # Add food (from dead snake)
PACKET_FOOD_ADD_F = ord('f')    # Add food (variant)
PACKET_FOOD_EAT = ord('c')      # Food eaten/removed
PACKET_FAM_UPDATE = ord('h')    # Fullness update
PACKET_TAIL_REMOVE = ord('r')   # Remove tail segment
PACKET_DEATH = ord('v')         # Snake died (kill event)
PACKET_SECTOR_ON = ord('w')     # Sector active
PACKET_SECTOR_OFF = ord('W')    # Sector inactive
PACKET_MINIMAP = ord('u')       # Minimap data
PACKET_LEADERBOARD = ord('l')   # Leaderboard
PACKET_PREINIT = ord('6')       # Pre-init JS challenge
PACKET_PONG = ord('p')          # Pong response from server
PACKET_SNAKE_REMOVE_DEAD = ord('j')  # Remove dead snake completely

# Rotation packet set for quick lookup
ROTATION_PACKETS = {
    PACKET_ROTATE_E, PACKET_ROTATE_E2,
    PACKET_ROTATE_3, PACKET_ROTATE_4, PACKET_ROTATE_5,
}

# Movement packet set
MOVEMENT_PACKETS = {
    PACKET_MOVE_ABS, PACKET_MOVE_ABS2,
    PACKET_MOVE_GROW, PACKET_MOVE_GROW2,
}

# Food packet set
FOOD_ADD_PACKETS = {
    PACKET_FOOD_ADD, PACKET_FOOD_ADD_B, PACKET_FOOD_ADD_F,
}
