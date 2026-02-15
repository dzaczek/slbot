"""
CDP WebSocket Interceptor for slither.io.

Connects to Chrome DevTools Protocol via a direct WebSocket to localhost.
Intercepts game WebSocket frames, parses them in Python, and maintains
a GameState object. Actions are sent via CDP Runtime.evaluate.

Architecture:
    slither.io server ↔ Chrome (handles anti-bot) ↔ CDP WS ↔ Python

This gives us:
- get_game_data(): instant (reads from Python memory, no execute_script)
- send_action(): ~1-2ms via CDP Runtime.evaluate (vs ~10-15ms Selenium)
- Real-time state updates pushed via CDP events (no polling)
"""

import json
import time
import base64
import threading
import logging
import math
import urllib.request

import websocket as ws_lib

from ws_engine import GameState, SlitherWSClient
from ws_protocol import (
    PacketReader, TWO_PI,
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


def log(msg):
    print(msg, flush=True)


class CDPInterceptor:
    """
    Intercepts slither.io WebSocket traffic via Chrome DevTools Protocol.

    Usage:
        interceptor = CDPInterceptor(selenium_driver)
        interceptor.start()  # Starts listening for game WS frames

        data = interceptor.get_game_data()  # Instant, from memory
        interceptor.send_action(angle, boost)  # Via CDP Runtime.evaluate
    """

    MAX_FOODS = 300
    MAX_ENEMIES = 50
    MAX_BODY_PTS = 150

    def __init__(self, driver):
        """
        Args:
            driver: Selenium WebDriver instance (Chrome, must have
                    --remote-debugging-port and --remote-allow-origins=*)
        """
        self.driver = driver
        self.state = GameState()
        self._lock = threading.Lock()
        self._cdp_ws = None
        self._listener_thread = None
        self._running = False
        self._game_ws_request_id = None
        self._cdp_id_counter = 0
        self._want_etm_s = True  # slither.io uses timestamp mode
        self._packet_handler = SlitherWSClient.__new__(SlitherWSClient)
        # Initialize the handler's state to point to OUR state
        self._packet_handler.state = self.state
        self._packet_handler._lock = self._lock
        self._packet_handler._want_etm_s = True
        self._packet_handler._login_sent = False
        self._packet_handler._init_received = threading.Event()
        self._packet_handler._spawn_received = threading.Event()
        self._packet_handler._connected_event = threading.Event()
        self._packet_handler._close_requested = False
        self._packet_handler.nickname = "CDPBot"
        # CDP is passive — Chrome handles WS sends, so _send_binary is a no-op
        self._packet_handler._send_binary = lambda data: None
        self._packet_handler.ws = None
        self._init_received = self._packet_handler._init_received
        self._frames_received = 0

    def _next_id(self):
        self._cdp_id_counter += 1
        return self._cdp_id_counter

    def start(self):
        """Connect to Chrome DevTools and start intercepting WS frames."""
        if self._running:
            return

        # Get Chrome debugger address
        debugger_addr = self.driver.capabilities.get(
            'goog:chromeOptions', {}
        ).get('debuggerAddress', '')

        if not debugger_addr:
            log("[CDP] ERROR: No debuggerAddress in driver capabilities. "
                "Add --remote-debugging-port=0 --remote-allow-origins=* to Chrome options.")
            return

        # Get page target's DevTools WS URL
        try:
            resp = urllib.request.urlopen(f'http://{debugger_addr}/json').read()
            targets = json.loads(resp)
            page_target = next(
                (t for t in targets if t.get('type') == 'page'), None
            )
            if not page_target:
                log("[CDP] ERROR: No page target found")
                return
            ws_url = page_target['webSocketDebuggerUrl']
        except Exception as e:
            log(f"[CDP] ERROR: Failed to get DevTools URL: {e}")
            return

        # Connect to Chrome DevTools via WebSocket
        try:
            self._cdp_ws = ws_lib.WebSocket()
            self._cdp_ws.connect(ws_url)
            log(f"[CDP] Connected to Chrome DevTools")
        except Exception as e:
            log(f"[CDP] ERROR: Failed to connect to DevTools WS: {e}")
            return

        # Enable Network domain to capture WS frames
        self._cdp_send('Network.enable', {})
        log("[CDP] Network monitoring enabled")

        # Start listener thread
        self._running = True
        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="cdp-listener",
        )
        self._listener_thread.start()
        log("[CDP] Listener thread started")

    def stop(self):
        """Stop the interceptor."""
        self._running = False
        if self._cdp_ws:
            try:
                self._cdp_ws.close()
            except Exception:
                pass
            self._cdp_ws = None
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3.0)

    def reset(self):
        """Reset game state (called on force_restart/reconnect)."""
        with self._lock:
            old_connected = self.state.connected
            self.state = GameState()
            self.state.connected = old_connected
            self._packet_handler.state = self.state
            self._game_ws_request_id = None
            self._frames_received = 0
            self._init_received.clear()

    def _cdp_send(self, method, params=None):
        """Send a CDP command and return the response."""
        if not self._cdp_ws:
            return None
        msg_id = self._next_id()
        msg = {'id': msg_id, 'method': method, 'params': params or {}}
        try:
            self._cdp_ws.send(json.dumps(msg))
            # Read response (with timeout)
            self._cdp_ws.settimeout(5.0)
            while True:
                resp = json.loads(self._cdp_ws.recv())
                if resp.get('id') == msg_id:
                    return resp.get('result')
                # It's an event — process it
                self._handle_cdp_event(resp)
        except Exception as e:
            logger.debug(f"[CDP] Send error: {e}")
            return None

    def _cdp_send_fire_and_forget(self, method, params=None):
        """Send a CDP command without waiting for response."""
        if not self._cdp_ws:
            return
        msg_id = self._next_id()
        msg = {'id': msg_id, 'method': method, 'params': params or {}}
        try:
            self._cdp_ws.send(json.dumps(msg))
        except Exception:
            pass

    def _listen_loop(self):
        """Background thread: listen for CDP events."""
        self._cdp_ws.settimeout(0.1)
        while self._running:
            try:
                raw = self._cdp_ws.recv()
                if raw:
                    msg = json.loads(raw)
                    self._handle_cdp_event(msg)
            except ws_lib.WebSocketTimeoutException:
                continue
            except ws_lib.WebSocketConnectionClosedException:
                log("[CDP] DevTools connection closed")
                self._running = False
                break
            except Exception as e:
                if self._running:
                    logger.debug(f"[CDP] Listener error: {e}")
                continue

    def _handle_cdp_event(self, msg):
        """Process a CDP event message."""
        method = msg.get('method', '')

        if method == 'Network.webSocketCreated':
            url = msg['params'].get('url', '')
            rid = msg['params'].get('requestId', '')
            # Detect the game WebSocket (port 444 or /slither path)
            if '/slither' in url or ':444' in url:
                self._game_ws_request_id = rid
                self._frames_received = 0  # Reset frame count for new WS
                log(f"[CDP] Game WebSocket detected: {url} (rid={rid})")

        elif method == 'Network.webSocketFrameReceived':
            rid = msg['params'].get('requestId', '')
            if rid == self._game_ws_request_id:
                response = msg['params'].get('response', {})
                opcode = response.get('opcode', 2)
                payload_data = response.get('payloadData', '')
                if payload_data:
                    try:
                        if opcode == 2:
                            # Binary frame — base64 encoded
                            raw = base64.b64decode(payload_data)
                        else:
                            # Text frame — raw UTF-8 bytes
                            raw = payload_data.encode('utf-8')
                        self._handle_game_frame(raw)
                        self._frames_received += 1
                        if self._frames_received <= 3:
                            log(f"[CDP] Frame #{self._frames_received}: {len(raw)}B opcode={opcode} "
                                f"first_bytes={list(raw[:8])}")
                    except Exception as e:
                        logger.debug(f"[CDP] Frame decode error: {e}")

        elif method == 'Network.webSocketClosed':
            rid = msg['params'].get('requestId', '')
            if rid == self._game_ws_request_id:
                log(f"[CDP] Game WebSocket closed (rid={rid}, frames={self._frames_received})")
                with self._lock:
                    self.state.dead = True
                    self.state.playing = False

    def _handle_game_frame(self, data: bytes):
        """Parse a raw game WebSocket frame using the existing protocol handler."""
        if len(data) < 1:
            return

        # Reuse SlitherWSClient's message parser (handles framing + dispatch)
        try:
            self._packet_handler._handle_message(data)
        except Exception as e:
            logger.debug(f"[CDP] Packet parse error: {e}")

        # Track that init has been received (snake identification happens on main thread)

    def try_activate(self):
        """Try to activate game state by identifying our snake. Call from main thread."""
        if self.state.playing:
            return True
        if not self._init_received.is_set() or self._frames_received < 10:
            return False

        # Identify our snake by matching browser position to parsed WS snakes
        try:
            result = self.driver.execute_script(
                "if(!window.slither) return null;"
                "return {x: window.slither.xx, y: window.slither.yy};"
            )
            if result and result.get('x') is not None:
                sx, sy = float(result['x']), float(result['y'])
                with self._lock:
                    best_id, best_dist = -1, float('inf')
                    for sid, snake in self.state.snakes.items():
                        d = (snake.x - sx) ** 2 + (snake.y - sy) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_id = sid
                    if best_id != -1 and best_dist < 10000:  # Within 100 units
                        self.state.my_id = best_id
                        self.state.dead = False
                        self.state.playing = True
                        self.state.connected = True
                        log(f"[CDP] Game state active — snake id={best_id} "
                            f"(dist={best_dist**.5:.1f}, {len(self.state.snakes)} snakes, "
                            f"{self._frames_received} frames)")
                        return True
        except Exception as e:
            logger.debug(f"[CDP] Could not identify snake: {e}")
        return False

    # ─── Public API (browser_engine compatible) ─────────────────────

    @property
    def active(self):
        """True if interceptor is running, receiving frames, and has valid game state."""
        return (self._running and
                self._game_ws_request_id is not None and
                self._frames_received > 0 and
                self.state.playing)

    def get_game_data(self):
        """
        Returns game state in browser_engine-compatible dict format.
        Instant: reads from Python memory, no Selenium round-trip.
        """
        with self._lock:
            return self._packet_handler._build_game_data()

    def send_action(self, angle, boost):
        """
        Send steering command via CDP Runtime.evaluate.
        Much faster than Selenium execute_script (~1-2ms vs ~10-15ms).
        """
        is_boost = 1 if boost > 0.5 else 0
        js = (
            f"if(window.slither){{"
            f"var c=document.getElementById('mc')||document.querySelector('canvas');"
            f"var cx=c?c.width/2:400,cy=c?c.height/2:300;"
            f"window.xm=cx+Math.cos({angle})*300;"
            f"window.ym=cy+Math.sin({angle})*300;"
            f"if(typeof window.mx!=='undefined')window.mx=window.xm;"
            f"if(typeof window.my!=='undefined')window.my=window.ym;"
            f"window.accelerating={'true' if is_boost else 'false'};"
            f"if(window.setAcceleration)window.setAcceleration({is_boost});"
            f"}}"
        )
        self._cdp_send_fire_and_forget('Runtime.evaluate', {
            'expression': js,
            'returnByValue': False,
        })

    def send_action_get_data(self, angle, boost):
        """Combined send + read. Action via CDP, state from memory."""
        self.send_action(angle, boost)
        return self.get_game_data()
