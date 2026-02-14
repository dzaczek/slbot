"""
Native WebSocket client for slither.io.

Connects directly to the game server via WebSocket, bypassing the browser.
Maintains a full GameState in memory, updated by parsing binary protocol packets.

Usage:
    client = SlitherWSClient("ws://1.2.3.4:444/slither", nickname="Bot")
    if client.connect():
        client.send_angle(1.57)
        data = client.get_game_data()  # Same format as browser_engine
        client.disconnect()
"""

import math
import time
import struct
import subprocess
import threading
import logging
import json

try:
    import websocket  # websocket-client library
except ImportError:
    websocket = None

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from ws_protocol import (
    PacketReader, TWO_PI,
    decode_angle, decode_angle_16, decode_speed, decode_fam, decode_relative_pos,
    encode_angle, encode_angle_precise, encode_setup_request, encode_start_login,
    encode_login, encode_boost_start, encode_boost_stop,
    PACKET_INIT, PACKET_SNAKE_ADD, PACKET_PREINIT, PACKET_PONG,
    PACKET_MOVE_ABS, PACKET_MOVE_ABS2, PACKET_MOVE_GROW, PACKET_MOVE_GROW2,
    PACKET_ROTATE_E, PACKET_ROTATE_E2, PACKET_ROTATE_3, PACKET_ROTATE_4, PACKET_ROTATE_5,
    PACKET_FOOD_ADD, PACKET_FOOD_ADD_B, PACKET_FOOD_ADD_F,
    PACKET_FOOD_EAT, PACKET_FAM_UPDATE, PACKET_TAIL_REMOVE,
    PACKET_DEATH, PACKET_SECTOR_ON, PACKET_SECTOR_OFF,
    PACKET_MINIMAP, PACKET_LEADERBOARD, PACKET_SNAKE_REMOVE_DEAD,
    ROTATION_PACKETS, MOVEMENT_PACKETS, FOOD_ADD_PACKETS,
)

logger = logging.getLogger(__name__)


# ─── Data classes ────────────────────────────────────────────────────────

@dataclass
class SnakePoint:
    """A single body segment of a snake."""
    x: float
    y: float
    dying: bool = False


@dataclass
class Snake:
    """Represents a snake in the game."""
    id: int
    x: float = 0.0
    y: float = 0.0
    ang: float = 0.0       # Current angle (radians)
    wang: float = 0.0      # Target/wanted angle
    sp: float = 5.78       # Speed (default normal speed)
    sc: float = 1.0        # Scale factor
    fam: float = 0.0       # Fullness
    skin: int = 0
    name: str = ""
    pts: List[SnakePoint] = field(default_factory=list)
    alive: bool = True
    # Interpolation fields
    eang: float = 0.0      # Extrapolated angle
    tsp: float = 0.0       # Target speed


@dataclass
class Food:
    """A food item on the map."""
    id: int
    x: float
    y: float
    size: float = 1.0
    color: int = 0


@dataclass
class GameState:
    """Full game state maintained from WebSocket packets."""
    my_id: int = -1
    snakes: Dict[int, Snake] = field(default_factory=dict)
    foods: Dict[int, Food] = field(default_factory=dict)

    # Map config (from packet 'a')
    grd: float = 21600.0       # Grid/map radius
    map_radius: float = 21600.0
    map_center_x: float = 21600.0
    map_center_y: float = 21600.0

    # Physics config
    spangdv: float = 4.8
    nsp1: float = 4.25
    nsp2: float = 0.5
    nsp3: float = 12.0
    mamu: float = 0.033
    mamu2: float = 0.028
    cst: float = 0.43
    fpsls_default: float = 0.0
    fmlts_default: float = 0.0

    # Zoom
    gsc: float = 0.9

    # State flags
    dead: bool = False
    connected: bool = False
    playing: bool = False

    # Leaderboard
    leaderboard: List[Tuple[str, int]] = field(default_factory=list)

    # Protocol config received
    protocol_version: int = 0


# ─── WebSocket Client ────────────────────────────────────────────────────

class SlitherWSClient:
    """Native WebSocket client for slither.io."""

    # Limits matching browser_engine
    MAX_FOODS = 300
    MAX_ENEMIES = 50
    MAX_BODY_PTS = 150

    def __init__(self, server_url: str, nickname: str = "dzaczekAI"):
        if websocket is None:
            raise ImportError("websocket-client is required. Install: pip install websocket-client")

        self.server_url = server_url
        self.nickname = nickname
        self.ws: Optional[websocket.WebSocketApp] = None
        self.state = GameState()
        self._lock = threading.Lock()
        self._ws_thread: Optional[threading.Thread] = None
        self._connected_event = threading.Event()
        self._init_received = threading.Event()
        self._spawn_received = threading.Event()
        self._close_requested = False
        self._want_etm_s = False
        self._last_ping = 0.0
        self._ping_interval = 5.0  # seconds
        self._reconnect_count = 0
        self._max_reconnects = 5
        self._login_sent = False

    # ─── Connection lifecycle ────────────────────────────────────────

    def connect(self, timeout: float = 15.0) -> bool:
        """
        Connect to the slither.io server.
        Handles the pre-init challenge, login, and waits for spawn.
        Returns True if successfully connected and spawned.
        """
        self._close_requested = False
        self._login_sent = False
        self._want_etm_s = False
        self.state = GameState()
        self._connected_event.clear()
        self._init_received.clear()
        self._spawn_received.clear()

        logger.info(f"[WS] Connecting to {self.server_url}...")

        try:
            self.ws = websocket.WebSocketApp(
                self.server_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                header={
                    "Origin": "http://slither.com",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/144.0.0.0 Safari/537.36",
                },
            )

            self._ws_thread = threading.Thread(
                target=self._run_ws,
                daemon=True,
                name="ws-receiver",
            )
            self._ws_thread.start()

            # Wait for connection (triggers _on_open which sends 'c')
            if not self._connected_event.wait(timeout=timeout):
                logger.error("[WS] Connection timeout")
                self.disconnect()
                return False

            # Wait for init packet 'a' (comes after challenge '6' + login)
            if not self._init_received.wait(timeout=timeout):
                logger.error("[WS] Init packet timeout — challenge or login may have failed")
                self.disconnect()
                return False

            logger.info(f"[WS] Init received. Protocol version: {self.state.protocol_version}")

            # Wait for our snake to spawn
            if not self._spawn_received.wait(timeout=timeout):
                logger.warning("[WS] Spawn timeout — may not have received snake packet yet")
                # Don't disconnect — some servers are slow to spawn
                # The bot will detect dead=True and retry

            self.state.connected = True
            self.state.playing = True
            self.state.dead = False
            logger.info(f"[WS] Connected! My snake ID: {self.state.my_id}")
            return True

        except Exception as e:
            logger.error(f"[WS] Connection failed: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        """Clean disconnect."""
        self._close_requested = True
        self.state.connected = False
        self.state.playing = False
        if self.ws:
            try:
                self.ws.close()
            except Exception:
                pass
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3.0)
        self.ws = None
        logger.info("[WS] Disconnected")

    def reconnect(self, timeout: float = 15.0) -> bool:
        """Disconnect and reconnect (used after death)."""
        self.disconnect()
        time.sleep(0.5)
        self._reconnect_count += 1
        return self.connect(timeout=timeout)

    def _run_ws(self):
        """Run WebSocket event loop in background thread."""
        try:
            self.ws.run_forever(
                ping_interval=0,  # We handle pings ourselves
                ping_timeout=None,
                skip_utf8_validation=True,
            )
        except Exception as e:
            if not self._close_requested:
                logger.error(f"[WS] Event loop error: {e}")

    def _send_binary(self, data: bytes):
        """Send binary data to server."""
        if self.ws and self.state.connected or self._connected_event.is_set():
            try:
                self.ws.send(data, opcode=websocket.ABNF.OPCODE_BINARY)
            except Exception as e:
                logger.error(f"[WS] Send error: {e}")

    # ─── WebSocket callbacks ─────────────────────────────────────────

    def _on_open(self, ws):
        self._connected_event.set()
        self._last_ping = time.time()
        self._want_etm_s = True
        logger.info("[WS] WebSocket opened — sending setup request + cstr")
        # Step 1: Send [1] — request timestamp mode (want_etm_s)
        self._send_binary(encode_setup_request())
        # Step 2: Send [99, 0] — cstr 'c' + null terminator
        self._send_binary(encode_start_login())

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        if isinstance(message, str):
            # Text message — shouldn't happen in normal protocol
            logger.debug(f"[WS] Text message: {message[:100]}")
            return

        if not message or len(message) == 0:
            return

        try:
            self._handle_message(message)
        except Exception as e:
            logger.debug(f"[WS] Message parse error: {e} (len={len(message)}, first={message[0] if message else '?'})")

        # Periodic ping
        now = time.time()
        if now - self._last_ping > self._ping_interval:
            self._last_ping = now
            try:
                self._send_binary(bytes([251]))
            except Exception:
                pass

    def _on_error(self, ws, error):
        if not self._close_requested:
            logger.error(f"[WS] Error: {error}")

    def _on_close(self, ws, close_status, close_msg):
        if not self._close_requested:
            logger.warning(f"[WS] Connection closed: {close_status} {close_msg}")
        self.state.connected = False

    # ─── Message dispatcher ──────────────────────────────────────────

    def _handle_message(self, data: bytes):
        """
        Parse and dispatch a binary message from the server.

        slither.io framing (from game JS ws.onmessage):
        1. If want_etm_s: skip 2-byte elapsed time header, else m=0
        2. Sub-packet framing:
           - If byte < 32: 2-byte big-endian length (byte<<8 | next_byte), then payload
           - If byte >= 32: 1-byte length = byte - 32, then payload (OR single packet if first byte)
        3. If first data byte >= 32: entire message (after etm skip) is one packet
        """
        if len(data) < 1:
            return

        a = data
        m = 0

        # Skip 2-byte elapsed time header if we requested timestamps
        if getattr(self, '_want_etm_s', False) and len(a) >= 2:
            m = 2

        if m >= len(a):
            return

        # Check if this is framed (sub-packets) or a single packet
        if a[m] < 32:
            # Framed: multiple sub-packets
            while m < len(a):
                if a[m] < 32:
                    if m + 1 >= len(a):
                        break
                    pkt_len = (a[m] << 8) | a[m + 1]
                    m += 2
                else:
                    pkt_len = a[m] - 32
                    m += 1
                if m + pkt_len > len(a):
                    break
                sub = a[m:m + pkt_len]
                m += pkt_len
                self._dispatch_packet(sub)
        else:
            # Single packet: the rest of the data after the header
            sub = a[m:]
            self._dispatch_packet(sub)

    def _dispatch_packet(self, pkt: bytes):
        """Dispatch a single decoded packet. First byte is the packet type."""
        if len(pkt) < 1:
            return

        ptype = pkt[0]

        with self._lock:
            if ptype == PACKET_PREINIT:
                self._handle_preinit_raw(pkt)
            elif ptype == PACKET_INIT:
                self._handle_init(pkt)
            elif ptype == PACKET_SNAKE_ADD:
                self._handle_snake_add(pkt)
            elif ptype in MOVEMENT_PACKETS:
                self._handle_movement(pkt, ptype)
            elif ptype in ROTATION_PACKETS:
                self._handle_rotation(pkt, ptype)
            elif ptype in FOOD_ADD_PACKETS:
                self._handle_food_add(pkt, ptype)
            elif ptype == PACKET_FOOD_EAT:
                self._handle_food_eat(pkt)
            elif ptype == PACKET_FAM_UPDATE:
                self._handle_fam_update(pkt)
            elif ptype == PACKET_TAIL_REMOVE:
                self._handle_tail_remove(pkt)
            elif ptype == PACKET_DEATH:
                self._handle_death(pkt)
            elif ptype == PACKET_SNAKE_REMOVE_DEAD:
                self._handle_snake_remove(pkt)
            elif ptype == PACKET_LEADERBOARD:
                self._handle_leaderboard(pkt)
            elif ptype in (PACKET_SECTOR_ON, PACKET_SECTOR_OFF):
                pass
            elif ptype == PACKET_MINIMAP:
                pass
            elif ptype == PACKET_PONG:
                pass
            else:
                logger.debug(f"[WS] Unknown packet type: {ptype} ({chr(ptype) if 32 <= ptype < 127 else '?'}) len={len(pkt)}")

    # ─── Protocol handlers ───────────────────────────────────────────

    def _handle_preinit_raw(self, pkt: bytes):
        """
        Handle pre-init packet '6' — server version / challenge.

        From game JS (gotServerVersion in game1107241958.js):
        1. The payload is a "server version" string (ASCII A-z chars, codes 65-122)
        2. The client validates it with isValidVersion()
        3. The client generates a 27-byte random_id
        4. Sends random_id bytes, then the login packet (lgba)

        The random_id generation (exact JS):
            alpha_chars = "abcdefghijklmnopqrstuvwxyz"  // 26 chars
            for (var i=0; i<27; i++)
                random_id += String.fromCharCode(
                    65 + (Math.random()<.5 ? 0 : 32)
                    + alpha_chars.charCodeAt(i)     // NaN for i=26!
                    + Math.floor(Math.random()*26)
                );
            // For i=26: charCodeAt(26)=NaN, so fromCharCode(NaN)=chr(0), byte=0
        """
        import random as _random

        version_data = pkt[1:]
        version_str = version_data.decode('ascii', errors='replace')
        logger.info(f"[WS] Server version: {version_str[:30]}... ({len(version_data)} bytes)")

        # Validate server version (all chars must be in range [65, 122])
        valid = all(65 <= b <= 122 for b in version_data) if version_data else False
        if not valid:
            logger.warning("[WS] Invalid server version — proceeding anyway")

        # Generate random_id (27 bytes) — exact match to game JS algorithm
        alpha_chars = "abcdefghijklmnopqrstuvwxyz"  # 26 chars (indices 0-25)
        random_id = bytearray(27)
        for i in range(27):
            if i < 26:
                case_offset = 0 if _random.random() < 0.5 else 32
                char_code = ord(alpha_chars[i])
                rand_offset = _random.randint(0, 25)
                random_id[i] = (65 + case_offset + char_code + rand_offset) & 0xFF
            else:
                # i=26: alpha_chars.charCodeAt(26) = NaN in JS → String.fromCharCode(NaN) = chr(0)
                random_id[i] = 0

        # Send the random_id
        self._send_binary(bytes(random_id))
        logger.info(f"[WS] Challenge response sent (27 bytes)")

        # Send login packet immediately after (game JS sends lgba right after idba)
        login_data = encode_login(self.nickname, skin_id=_random.randint(0, 8))
        self._send_binary(login_data)
        logger.info(f"[WS] Login sent: {self.nickname} ({len(login_data)} bytes)")
        self._login_sent = True

    def _handle_init(self, pkt: bytes):
        """
        Handle init packet 'a' — game configuration.

        Format (raw): [type:1][grd:3][...config values...]
        Contains: grd, spangdv, nsp1, nsp2, nsp3, mamu, mamu2, cst, protocol_version, etc.
        """
        r = PacketReader(pkt, 1)  # Skip type byte

        if r.remaining < 3:
            return

        # Parsing matches the deobfuscated game JS exactly:
        # grd = a[m]<<16 | a[m+1]<<8 | a[m+2]   (3 bytes)
        grd = r.read_uint24()
        self.state.grd = grd
        self.state.map_center_x = float(grd)
        self.state.map_center_y = float(grd)

        # nmscps (2), sector_size (2), sector_count (2)
        if r.remaining >= 6:
            r.skip(2)  # nmscps
            r.skip(2)  # sector_size
            r.skip(2)  # sector_count_along_edge

        # spangdv = a[m]/10  (1 byte)
        if r.remaining >= 1:
            self.state.spangdv = r.read_uint8() / 10.0

        # nsp1, nsp2, nsp3 = uint16/100 each (2 bytes each)
        if r.remaining >= 6:
            self.state.nsp1 = r.read_uint16() / 100.0
            self.state.nsp2 = r.read_uint16() / 100.0
            self.state.nsp3 = r.read_uint16() / 100.0

        # mamu, mamu2 = uint16/1000 each (2 bytes each)
        if r.remaining >= 4:
            self.state.mamu = r.read_uint16() / 1000.0
            self.state.mamu2 = r.read_uint16() / 1000.0

        # cst = uint16/1000 (2 bytes)
        if r.remaining >= 2:
            self.state.cst = r.read_uint16() / 1000.0

        # protocol_version (1 byte)
        if r.remaining >= 1:
            self.state.protocol_version = r.read_uint8()

        # default_msl (1 byte) — skip
        if r.remaining >= 1:
            r.skip(1)

        # real_sid (2 bytes) — skip
        if r.remaining >= 2:
            r.skip(2)

        # flux_grd (3 bytes) — the actual playable radius
        flux_grd = grd * 0.98  # default
        if r.remaining >= 3:
            flux_grd = r.read_uint24()

        self.state.map_radius = float(flux_grd)

        # If cst indicates a smaller map
        if self.state.cst > 0.1 and self.state.cst < 0.95 and flux_grd == grd * 0.98:
            self.state.map_radius = grd * self.state.cst

        logger.info(f"[WS] Init: grd={grd} flux_grd={flux_grd} map_radius={self.state.map_radius:.0f} "
                     f"center=({self.state.map_center_x:.0f},{self.state.map_center_y:.0f}) "
                     f"proto={self.state.protocol_version} cst={self.state.cst:.3f}")

        self._init_received.set()

    def _handle_snake_add(self, pkt: bytes):
        """
        Handle packet 's' — add or update a snake.

        Long format (add): [type:1][id:2][ang:3][unused:1][x:3][y:3]
                           [fam:3][skin:1][nameLen:1][name:nameLen][body_pts...]
        Short format (remove): [type:1][id:2] (dlen == 2)
        """
        r = PacketReader(pkt, 1)  # Skip type byte

        if r.remaining < 2:
            return

        snake_id = r.read_uint16()

        # Short packet = snake removed
        if r.remaining < 4:
            if snake_id in self.state.snakes:
                del self.state.snakes[snake_id]
            return

        # Long packet = add snake
        try:
            # Angle: 3 bytes → uint24 → angle
            ang_raw = r.read_uint24()
            ang = ang_raw * TWO_PI / 16777215.0

            # 1 byte unused/ehang direction
            r.skip(1)

            # Position: 3 bytes each (int24) — these are absolute * 5
            # Actually in standard protocol: x and y as int24 / 5
            # But commonly they're just absolute int24
            snake_x = r.read_int24() / 5.0 if r.remaining >= 3 else 0.0
            snake_y = r.read_int24() / 5.0 if r.remaining >= 3 else 0.0

            # Fullness: uint24
            fam_raw = r.read_uint24() if r.remaining >= 3 else 0
            fam = decode_fam(fam_raw)

            # Skin: uint8
            skin = r.read_uint8() if r.remaining >= 1 else 0

            # Name: length-prefixed string
            name = ""
            if r.remaining >= 1:
                name_len = r.read_uint8()
                if r.remaining >= name_len and name_len > 0:
                    name = r.read_bytes(name_len).decode('utf-8', errors='replace')

            snake = Snake(
                id=snake_id,
                x=snake_x,
                y=snake_y,
                ang=ang,
                wang=ang,
                fam=fam,
                skin=skin,
                name=name,
            )

            # Calculate scale from fam
            snake.sc = min(6.0, 1.0 + (fam / 0.2) * 0.1)

            # Read body points
            # Body points come as pairs of int24/5 (absolute positions)
            while r.remaining >= 4:
                # Each body point: 2 bytes (relative to last point, or absolute)
                # In standard protocol: body points are absolute int16 pairs
                if r.remaining >= 4:
                    px = r.read_int16()
                    py = r.read_int16()
                    # Points are stored as absolute / some factor
                    # In some implementations they're relative + accumulated
                    snake.pts.append(SnakePoint(x=px / 5.0, y=py / 5.0))
                else:
                    break

            # If no body points were parsed, create a single point at head
            if not snake.pts:
                snake.pts.append(SnakePoint(x=snake_x, y=snake_y))

            self.state.snakes[snake_id] = snake

            # Detect if this is our snake (first snake added after login)
            if self.state.my_id == -1 and name == self.nickname:
                self.state.my_id = snake_id
                self.state.dead = False
                self._spawn_received.set()
                logger.info(f"[WS] My snake spawned: id={snake_id} pos=({snake_x:.0f},{snake_y:.0f})")
            elif self.state.my_id == -1 and not self._spawn_received.is_set():
                # If we haven't identified our snake yet, this might be it
                # (some servers don't echo the name back)
                self.state.my_id = snake_id
                self.state.dead = False
                self._spawn_received.set()
                logger.info(f"[WS] Assumed my snake: id={snake_id} pos=({snake_x:.0f},{snake_y:.0f})")

        except Exception as e:
            logger.debug(f"[WS] Snake add parse error: {e}")

    def _handle_movement(self, pkt: bytes, ptype: int):
        """
        Handle movement packets 'g', 'G', 'n', 'N'.

        g/G: Position update (absolute/relative)
        n/N: Position update + body grow (new point added)

        Format varies:
        g: [ts:2][type:1][id:2][x:2][y:2]           (absolute position, int16)
        G: [ts:2][type:1][id:2][dx:1][dy:1]          (relative, byte-128 / 2)
        n: [ts:2][type:1][id:2][x:2][y:2][fx:2][fy:2] (abs + new food point)
        N: [ts:2][type:1][id:2][dx:1][dy:1][fx:2][fy:2] (rel + new food point)
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 2:
            return

        snake_id = r.read_uint16()
        snake = self.state.snakes.get(snake_id)
        if not snake:
            return

        is_relative = ptype in (PACKET_MOVE_ABS2, PACKET_MOVE_GROW2)
        has_grow = ptype in (PACKET_MOVE_GROW, PACKET_MOVE_GROW2)

        if is_relative:
            # Relative movement: 1 byte each
            if r.remaining < 2:
                return
            dx = decode_relative_pos(r.read_uint8())
            dy = decode_relative_pos(r.read_uint8())
            snake.x += dx
            snake.y += dy
        else:
            # Absolute movement: int16 each, divided by 5
            if r.remaining < 4:
                return
            # In the standard protocol, positions in movement packets
            # are absolute int16 values (not divided by 5)
            abs_x = r.read_int16()
            abs_y = r.read_int16()
            # These are typically absolute coordinates / some factor
            # or straight absolute — depends on server variant
            snake.x = abs_x
            snake.y = abs_y

        if has_grow:
            # New body point (from eating food)
            if r.remaining >= 4:
                fx = r.read_int16()
                fy = r.read_int16()
                snake.pts.append(SnakePoint(x=float(fx), y=float(fy)))

        # Move head point
        if snake.pts:
            snake.pts[-1] = SnakePoint(x=snake.x, y=snake.y)

    def _handle_rotation(self, pkt: bytes, ptype: int):
        """
        Handle rotation packets 'e', 'E', '3', '4', '5'.

        e: [ts:2][type:1][id:2][ang:1][wang:1][sp:1]
        E: [ts:2][type:1][id:2][ang:2][wang:2][sp:1]
        3: [ts:2][type:1][id:2][ang:1][sp:1]
        4: [ts:2][type:1][id:2][ang:1][wang:1]
        5: [ts:2][type:1][id:2][ang:1]
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 2:
            return

        snake_id = r.read_uint16()
        snake = self.state.snakes.get(snake_id)
        if not snake:
            return

        if ptype == PACKET_ROTATE_E:
            # e: ang(1) + wang(1) + sp(1)
            if r.remaining >= 3:
                snake.ang = decode_angle(r.read_uint8())
                snake.wang = decode_angle(r.read_uint8())
                snake.sp = decode_speed(r.read_uint8())
        elif ptype == PACKET_ROTATE_E2:
            # E: ang(2) + wang(2) + sp(1) — higher precision
            if r.remaining >= 5:
                snake.ang = decode_angle_16(r.read_uint16())
                snake.wang = decode_angle_16(r.read_uint16())
                snake.sp = decode_speed(r.read_uint8())
        elif ptype == PACKET_ROTATE_3:
            # 3: ang(1) + sp(1)
            if r.remaining >= 2:
                snake.ang = decode_angle(r.read_uint8())
                snake.sp = decode_speed(r.read_uint8())
        elif ptype == PACKET_ROTATE_4:
            # 4: ang(1) + wang(1)
            if r.remaining >= 2:
                snake.ang = decode_angle(r.read_uint8())
                snake.wang = decode_angle(r.read_uint8())
        elif ptype == PACKET_ROTATE_5:
            # 5: ang(1)
            if r.remaining >= 1:
                snake.ang = decode_angle(r.read_uint8())

    def _handle_food_add(self, pkt: bytes, ptype: int):
        """
        Handle food add packets 'F', 'b', 'f'.

        F: [ts:2][type:1][color:1][x:2][y:2][sz:1]        (natural spawn)
        b: [ts:2][type:1][x:2][y:2][sz:1]                  (from dead snake, multiple items)
        f: [ts:2][type:1][color:1][x:2][y:2][sz:1]        (variant, eating leftovers)
        """
        r = PacketReader(pkt, 1)

        if ptype == PACKET_FOOD_ADD:
            # F: single food with color
            if r.remaining < 6:
                return
            color = r.read_uint8()
            x = r.read_int16()
            y = r.read_int16()
            sz = r.read_uint8() / 5.0 if r.remaining >= 1 else 1.0
            food_id = (x << 16) | (y & 0xFFFF)  # Use position as ID
            self.state.foods[food_id] = Food(id=food_id, x=float(x), y=float(y), size=max(0.2, sz), color=color)

        elif ptype == PACKET_FOOD_ADD_B:
            # b: multiple foods from dead snake
            while r.remaining >= 5:
                x = r.read_int16()
                y = r.read_int16()
                sz = r.read_uint8() / 5.0
                food_id = (x << 16) | (y & 0xFFFF)
                self.state.foods[food_id] = Food(id=food_id, x=float(x), y=float(y), size=max(0.2, sz))

        elif ptype == PACKET_FOOD_ADD_F:
            # f: similar to F
            if r.remaining < 6:
                return
            color = r.read_uint8()
            x = r.read_int16()
            y = r.read_int16()
            sz = r.read_uint8() / 5.0 if r.remaining >= 1 else 1.0
            food_id = (x << 16) | (y & 0xFFFF)
            self.state.foods[food_id] = Food(id=food_id, x=float(x), y=float(y), size=max(0.2, sz), color=color)

    def _handle_food_eat(self, pkt: bytes):
        """
        Handle food eaten/removed packet 'c'.

        Format: [ts:2][type:1][x:2][y:2][eater_id:2]
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 4:
            return

        x = r.read_int16()
        y = r.read_int16()
        food_id = (x << 16) | (y & 0xFFFF)
        self.state.foods.pop(food_id, None)

    def _handle_fam_update(self, pkt: bytes):
        """
        Handle fullness update packet 'h'.

        Format: [ts:2][type:1][id:2][fam:3]
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 5:
            return

        snake_id = r.read_uint16()
        fam_raw = r.read_uint24()
        fam = decode_fam(fam_raw)

        snake = self.state.snakes.get(snake_id)
        if snake:
            snake.fam = fam
            # Update scale from fam
            snake.sc = min(6.0, 1.0 + (fam / 0.2) * 0.1)

    def _handle_tail_remove(self, pkt: bytes):
        """
        Handle tail remove packet 'r'.

        Format: [ts:2][type:1][id:2]
        Removes the oldest body point from the snake.
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 2:
            return

        snake_id = r.read_uint16()
        snake = self.state.snakes.get(snake_id)
        if snake and snake.pts:
            snake.pts.pop(0)

    def _handle_death(self, pkt: bytes):
        """
        Handle death packet 'v'.

        Format: [ts:2][type:1][dead_id:2][killer_id:2]
        The dead snake drops food, the killer gets credit.
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 2:
            return

        dead_id = r.read_uint16()
        killer_id = r.read_uint16() if r.remaining >= 2 else -1

        # Check if our snake died
        if dead_id == self.state.my_id:
            self.state.dead = True
            self.state.playing = False
            logger.info(f"[WS] Our snake died! Killer ID: {killer_id}")

        # Mark snake as dead
        snake = self.state.snakes.get(dead_id)
        if snake:
            snake.alive = False

    def _handle_snake_remove(self, pkt: bytes):
        """Handle packet 'j' — remove snake entirely."""
        r = PacketReader(pkt, 1)
        if r.remaining < 2:
            return
        snake_id = r.read_uint16()
        self.state.snakes.pop(snake_id, None)

    def _handle_leaderboard(self, pkt: bytes):
        """
        Handle leaderboard packet 'l'.

        Format: [ts:2][type:1][count:1]
                then for each entry: [id:2][nameLen:1][name:nameLen][score:4]
        """
        r = PacketReader(pkt, 1)

        if r.remaining < 1:
            return

        # The format varies; some implementations use different layouts
        # For now, just note we received it
        pass

    # ─── Commands (Client -> Server) ─────────────────────────────────

    def send_angle(self, angle_rad: float):
        """Send steering angle to server."""
        data = encode_angle(angle_rad)
        self._send_binary(data)

    def send_boost(self, active: bool):
        """Send boost command to server."""
        if active:
            self._send_binary(encode_boost_start())
        else:
            self._send_binary(encode_boost_stop())

    # ─── State access (browser_engine compatible) ────────────────────

    def get_game_data(self) -> dict:
        """
        Returns game state in the EXACT same dict format as browser_engine.get_game_data().
        Thread-safe: acquires lock and copies data.
        """
        with self._lock:
            return self._build_game_data()

    def _build_game_data(self) -> dict:
        """Build game data dict (must be called under lock)."""

        if self.state.dead or not self.state.playing:
            return {'dead': True}

        my_snake = self.state.snakes.get(self.state.my_id)
        if not my_snake:
            return {'dead': True}

        mx, my = my_snake.x, my_snake.y

        # View radius calculation (matches browser: canvas / gsc)
        # Standard canvas 800x600, gsc ~0.9
        gsc = self.state.gsc
        view_width = 800.0 / gsc
        view_height = 600.0 / gsc
        view_radius = max(view_width, view_height) / 2.0

        # My snake body points
        my_pts = []
        pts_len = len(my_snake.pts)
        if pts_len > 0:
            # Trim ghost tail (same logic as browser_engine JS)
            trim_count = int(pts_len * 0.15) + int((my_snake.sp or 5.7) * 2.0)
            start_idx = min(pts_len - 1, trim_count)
            step = max(1, pts_len // self.MAX_BODY_PTS)
            for j in range(start_idx, pts_len, step):
                if len(my_pts) >= self.MAX_BODY_PTS:
                    break
                pt = my_snake.pts[j]
                my_pts.append([pt.x, pt.y])

        my_data = {
            'x': mx,
            'y': my,
            'ang': my_snake.ang,
            'sp': my_snake.sp,
            'sc': my_snake.sc,
            'len': pts_len,
            'pts': my_pts,
        }

        # Foods — filter by view radius, sort by distance, limit
        view_rad_sq = view_radius * view_radius * 1.2
        food_list = []
        for food in self.state.foods.values():
            dx = food.x - mx
            dy = food.y - my
            dist_sq = dx * dx + dy * dy
            if dist_sq < view_rad_sq:
                food_list.append((food.x, food.y, food.size, dist_sq))

        food_list.sort(key=lambda f: f[3])
        visible_foods = [[f[0], f[1], f[2]] for f in food_list[:self.MAX_FOODS]]

        # Enemies — filter by view radius, sort, limit
        search_rad_sq = view_radius * view_radius * 25  # 5x radius (matching browser)
        view_rad_sq_enemy = view_radius * view_radius * 2.0

        enemy_list = []
        for sid, snake in self.state.snakes.items():
            if sid == self.state.my_id:
                continue
            if not snake.alive:
                continue

            # Check if head or any body part is visible
            min_dist = float('inf')
            has_visible = False

            hdx = snake.x - mx
            hdy = snake.y - my
            head_dist = hdx * hdx + hdy * hdy
            if head_dist < view_rad_sq_enemy:
                has_visible = True
            min_dist = min(min_dist, head_dist)

            if not has_visible and snake.pts:
                pts_len_e = len(snake.pts)
                trim_e = int(pts_len_e * 0.1) + int((snake.sp or 5.7) * 2.0)
                start_e = min(pts_len_e - 1, trim_e)
                step_e = max(1, pts_len_e // 40)
                for j in range(start_e, pts_len_e, step_e):
                    pt = snake.pts[j]
                    bdx = pt.x - mx
                    bdy = pt.y - my
                    body_dist = bdx * bdx + bdy * bdy
                    if body_dist < view_rad_sq_enemy:
                        has_visible = True
                        break
                    min_dist = min(min_dist, body_dist)

            if has_visible or min_dist < search_rad_sq:
                priority = 0 if has_visible else 1
                enemy_list.append((snake, min_dist, priority))

        enemy_list.sort(key=lambda e: (e[2], e[1]))

        visible_enemies = []
        for snake, _, _ in enemy_list[:self.MAX_ENEMIES]:
            pts = []
            pts_len_e = len(snake.pts)
            if pts_len_e > 0:
                trim_e = int(pts_len_e * 0.15) + int((snake.sp or 5.7) * 2.0)
                start_e = min(pts_len_e - 1, trim_e)
                step_e = max(1, pts_len_e // self.MAX_BODY_PTS)
                for j in range(start_e, pts_len_e, step_e):
                    if len(pts) >= self.MAX_BODY_PTS:
                        break
                    pt = snake.pts[j]
                    pdx = pt.x - mx
                    pdy = pt.y - my
                    if pdx * pdx + pdy * pdy < search_rad_sq:
                        pts.append([pt.x, pt.y])

            visible_enemies.append({
                'id': snake.id,
                'x': snake.x,
                'y': snake.y,
                'ang': snake.ang,
                'sp': snake.sp,
                'sc': snake.sc,
                'pts': pts,
            })

        # Distance to wall
        dist_from_center = math.hypot(mx - self.state.map_center_x, my - self.state.map_center_y)
        dist_to_wall = self.state.map_radius - dist_from_center

        return {
            'dead': False,
            'self': my_data,
            'foods': visible_foods,
            'enemies': visible_enemies,
            'view_radius': view_radius,
            'gsc': gsc,
            'dist_to_wall': dist_to_wall,
            'dist_from_center': dist_from_center,
            'map_radius': self.state.map_radius,
            'map_center_x': self.state.map_center_x,
            'map_center_y': self.state.map_center_y,
            'boundary_type': 'circle',
            'boundary_vertices': [],
            'debug': {
                'total_slithers': len(self.state.snakes),
                'visible_enemies': len(visible_enemies),
                'total_foods': len(self.state.foods),
                'visible_foods': len(visible_foods),
                'dist_to_wall': round(dist_to_wall),
                'dist_from_center': round(dist_from_center),
                'snake_x': round(mx),
                'snake_y': round(my),
                'boundary_source': 'ws_grd',
                'boundary_type': 'circle',
                'map_vars': {
                    'grd': self.state.grd,
                    'map_radius': round(self.state.map_radius),
                    'map_center': f"{round(self.state.map_center_x)},{round(self.state.map_center_y)}",
                    'cst': self.state.cst,
                    'FINAL_source': 'ws_grd',
                    'FINAL_dist_to_wall': round(dist_to_wall),
                },
            },
        }


# ─── Server Discovery ────────────────────────────────────────────────────

# Cache discovered servers to avoid repeated lookups
_server_cache: Dict[str, List[dict]] = {}
_server_cache_time: Dict[str, float] = {}
_CACHE_TTL = 300.0  # 5 minutes


def discover_server(base_url: str = "http://slither.io", ws_override: str = "") -> str:
    """
    Discover the WebSocket server URL for slither.io.

    Priority:
    1. Direct override (from config/CLI)
    2. i33628.txt server list (standard slither.io)
    3. HTTP page scrape for WS URLs
    4. Selenium fallback (launch browser, extract bso.ip + bso.po)

    Returns: WebSocket URL like "ws://1.2.3.4:444/slither"
    """
    import random as _random

    # 1. Direct override
    if ws_override:
        logger.info(f"[WS] Using override server: {ws_override}")
        return ws_override

    # 2. Try i33628.txt server list (standard slither.io)
    try:
        servers = _fetch_server_list(base_url)
        if servers:
            # Pick a random server from the list
            server = _random.choice(servers)
            url = f"ws://{server['ip']}:{server['port']}/slither"
            logger.info(f"[WS] Discovered server via i33628.txt: {url} "
                        f"(from {len(servers)} servers)")
            return url
    except Exception as e:
        logger.debug(f"[WS] i33628.txt discovery failed: {e}")

    # 3. Try HTTP page scrape
    try:
        url = _discover_via_http(base_url)
        if url:
            return url
    except Exception as e:
        logger.debug(f"[WS] HTTP discovery failed: {e}")

    # 4. Selenium fallback
    try:
        url = _discover_via_selenium(base_url)
        if url:
            return url
    except Exception as e:
        logger.warning(f"[WS] Selenium discovery failed: {e}")

    raise RuntimeError(f"Could not discover WebSocket server for {base_url}")


def _decode_server_list(data: str) -> List[dict]:
    """
    Decode slither.io server list from i33628.txt content.

    The response is a character-encoded string where each server takes 22 characters
    (decoding to 11 bytes). Algorithm:
    1. Skip first char (version indicator)
    2. Convert chars to numbers: ord(c) - 97
    3. Apply index offset: v[i] -= 7*i
    4. Normalize: v[i] = (v[i] % 26 + 26) % 26
    5. Merge pairs into bytes: byte = v[2j]*16 + v[2j+1]
    6. Extract: IP(4) + Port(3) + AC(3) + CLU(1) = 11 bytes per server
    """
    if not data or len(data) < 23:
        return []

    # Skip version char
    data = data[1:]

    # Char -> number with index offset
    v = []
    for i, c in enumerate(data):
        val = ord(c) - 97
        val = val - 7 * i
        val = (val % 26 + 26) % 26
        v.append(val)

    # Merge pairs into bytes
    byte_count = len(v) // 2
    b = [v[2 * i] * 16 + v[2 * i + 1] for i in range(byte_count)]

    # Extract servers (11 bytes each)
    servers = []
    chunk_size = 11
    for i in range(len(b) // chunk_size):
        off = i * chunk_size
        ip = f"{b[off]}.{b[off+1]}.{b[off+2]}.{b[off+3]}"
        port = b[off + 4] * 65536 + b[off + 5] * 256 + b[off + 6]
        ac = b[off + 7] * 65536 + b[off + 8] * 256 + b[off + 9]
        clu = b[off + 10]

        # Validate: port should be reasonable
        if 1 <= port <= 65535:
            servers.append({
                'ip': ip,
                'port': port,
                'ac': ac,
                'clu': clu,
            })

    return servers


def _fetch_server_list(base_url: str) -> List[dict]:
    """Fetch and decode the slither.io server list from i33628.txt."""
    import urllib.request

    # Check cache
    cache_key = base_url
    if cache_key in _server_cache:
        age = time.time() - _server_cache_time.get(cache_key, 0)
        if age < _CACHE_TTL:
            return _server_cache[cache_key]

    # Determine the server list URL
    # Standard slither.io uses i33628.txt
    # Extract domain from base_url
    domain = base_url.replace('http://', '').replace('https://', '').rstrip('/')
    server_list_url = f"http://{domain}/i33628.txt"

    logger.info(f"[WS] Fetching server list from {server_list_url}...")

    req = urllib.request.Request(
        server_list_url,
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Origin': f'http://{domain}',
            'Referer': f'http://{domain}/',
        }
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode('ascii', errors='replace')

    servers = _decode_server_list(raw)

    if servers:
        _server_cache[cache_key] = servers
        _server_cache_time[cache_key] = time.time()
        logger.info(f"[WS] Decoded {len(servers)} servers from server list")
        for s in servers[:5]:
            logger.debug(f"  {s['ip']}:{s['port']} (ac={s['ac']}, clu={s['clu']})")

    return servers


def _discover_via_http(base_url: str) -> Optional[str]:
    """Try to discover server by scraping the game page for WS URLs."""
    import urllib.request
    import re

    try:
        req = urllib.request.Request(
            base_url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='replace')

        # Look for WebSocket URLs in the page JS
        ws_pattern = re.compile(r'wss?://[\d.]+:\d+/slither')
        matches = ws_pattern.findall(html)
        if matches:
            url = matches[0]
            logger.info(f"[WS] Discovered server via HTTP page scrape: {url}")
            return url

    except Exception as e:
        logger.debug(f"[WS] HTTP page scrape error: {e}")

    return None


def _discover_via_selenium(base_url: str) -> Optional[str]:
    """Launch headless browser once, connect to game, and extract WS server info."""
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--mute-audio")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=800,600")

        logger.info("[WS] Starting Selenium for server discovery...")
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        driver.get(base_url)
        time.sleep(5)

        # Extract server info from game's global objects
        ws_info = driver.execute_script("""
            // Try bso (best server object)
            if (window.bso && window.bso.ip && window.bso.po) {
                return {ip: window.bso.ip, port: window.bso.po};
            }
            // Try bso2 (WebSocket handle)
            if (window.bso2 && window.bso2.url) {
                return {url: window.bso2.url};
            }
            // Try sos (server list)
            if (window.sos && window.sos.length > 0) {
                var s = window.sos[0];
                return {ip: s.ip, port: s.po};
            }
            return null;
        """)

        driver.quit()

        if ws_info:
            if 'url' in ws_info:
                url = ws_info['url']
                if not url.endswith('/slither'):
                    url = url.rstrip('/') + '/slither'
            elif 'ip' in ws_info and 'port' in ws_info:
                url = f"ws://{ws_info['ip']}:{ws_info['port']}/slither"
            else:
                return None

            logger.info(f"[WS] Discovered server via Selenium: {url}")
            return url

    except Exception as e:
        logger.debug(f"[WS] Selenium discovery error: {e}")

    return None
