"""
WebSocket-based drop-in replacement for browser_engine.SlitherBrowser.

Provides the exact same API as the Selenium-based SlitherBrowser, but uses
a native WebSocket connection instead. This eliminates the browser overhead
(~500MB RAM/agent, ~30ms Selenium latency per call).

Usage:
    from browser_engine_ws import SlitherBrowser
    browser = SlitherBrowser(headless=True, nickname="Bot", base_url="http://slither.io")
    data = browser.get_game_data()
    browser.send_action(angle=1.5, boost=0)
    browser.close()
"""

import time
import logging

from ws_engine import SlitherWSClient, discover_server

logger = logging.getLogger(__name__)


def log(msg):
    print(msg, flush=True)


class SlitherBrowser:
    """
    Drop-in replacement for browser_engine.SlitherBrowser using WebSocket.
    All public methods match the Selenium version's signature and return format.
    """

    MAX_FOODS = 300
    MAX_ENEMIES = 50
    MAX_BODY_PTS = 150

    def __init__(self, headless=True, nickname="NEATBot", base_url="http://slither.io",
                 ws_server_url=""):
        """
        Initialize WebSocket-based browser engine.

        Args:
            headless: Ignored (no browser to show). Kept for API compatibility.
            nickname: Bot name sent to server.
            base_url: Game URL — used for server discovery if ws_server_url is empty.
            ws_server_url: Direct WebSocket URL override (e.g. "ws://1.2.3.4:444/slither").
        """
        self.nickname = nickname
        self.base_url = base_url
        self._ws_server_url = ws_server_url
        self._last_boost_state = False

        log(f"[WS-BROWSER] Initializing WebSocket backend for {base_url}...")

        # Discover server URL
        try:
            resolved_url = discover_server(base_url, ws_override=ws_server_url)
        except Exception as e:
            log(f"[WS-BROWSER] Server discovery failed: {e}")
            raise

        self.ws_client = SlitherWSClient(resolved_url, nickname=nickname)

        # Connect immediately
        log(f"[WS-BROWSER] Connecting to {resolved_url}...")
        if not self.ws_client.connect(timeout=20.0):
            log("[WS-BROWSER] Initial connection failed. Will retry on force_restart().")
        else:
            log("[WS-BROWSER] Connected!")

    def _handle_login(self):
        """
        Handles the initial login. For WebSocket, this is part of connect().
        Provided for API compatibility with browser_engine.
        """
        if not self.ws_client.state.connected:
            return self.ws_client.connect(timeout=15.0)
        return True

    def inject_override_script(self):
        """No-op: no browser to inject scripts into."""
        pass

    def scan_game_variables(self):
        """
        Returns game variables for diagnostic purposes.
        In WebSocket mode, we return what we know from the init packet.
        """
        state = self.ws_client.state
        return {
            'specific': {
                'grd': state.grd,
                'cst': state.cst,
                'protocol_version': state.protocol_version,
            },
            'numeric': {
                'grd': state.grd,
                'map_radius': state.map_radius,
            },
            'arrays': {},
            'snake_pos': {
                'x': state.snakes.get(state.my_id, None) and state.snakes[state.my_id].x or 0,
                'y': state.snakes.get(state.my_id, None) and state.snakes[state.my_id].y or 0,
            },
            'boundary_funcs': [],
        }

    def get_game_data(self):
        """
        Retrieves game state — same dict format as browser_engine.
        This is a fast operation: just reads from in-memory GameState.
        """
        if not self.ws_client.state.connected:
            return {'dead': True}

        try:
            return self.ws_client.get_game_data()
        except Exception as e:
            logger.debug(f"[WS-BROWSER] get_game_data error: {e}")
            return {'dead': True}

    def send_action(self, angle, boost):
        """
        Sends steering commands to the game.
        angle: target direction in radians
        boost: 0 or 1
        """
        if not self.ws_client.state.connected:
            return

        try:
            self.ws_client.send_angle(float(angle))

            # Only send boost state changes to minimize traffic
            is_boost = bool(boost > 0.5)
            if is_boost != self._last_boost_state:
                self.ws_client.send_boost(is_boost)
                self._last_boost_state = is_boost
        except Exception:
            pass

    def force_restart(self):
        """
        Resets the game state by reconnecting to the server.
        Much faster than browser_engine's page refresh (~0.5s vs ~5s).
        """
        log("[WS-BROWSER] Force restart: reconnecting...")
        self._last_boost_state = False

        try:
            if self.ws_client.reconnect(timeout=15.0):
                log("[WS-BROWSER] Reconnected successfully")
            else:
                log("[WS-BROWSER] Reconnect failed — will retry next call")
        except Exception as e:
            log(f"[WS-BROWSER] Reconnect error: {e}")

    def close(self):
        """Clean shutdown of WebSocket connection."""
        log("[WS-BROWSER] Closing...")
        try:
            self.ws_client.disconnect()
        except Exception:
            pass

    # ─── Overlay methods (no-op for WebSocket) ───────────────────────

    def inject_view_plus_overlay(self):
        """No-op: no browser canvas to draw on."""
        pass

    def update_view_plus_overlay(self, matrix=None, gsc=None, view_radius=None, debug_info=None):
        """No-op: no browser canvas to update."""
        pass
