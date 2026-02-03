import time
import json
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
# Assuming webdriver_manager is available, otherwise user might need to install it.
# from webdriver_manager.chrome import ChromeDriverManager 

class SlitherBrowser:
    """
    Manages the browser instance using Selenium.
    Handles 'JS Bridge' to communicate with Slither.io client.
    """
    def __init__(self, headless=False, nickname="NEATBot"):
        self.nickname = nickname
        self.options = Options()
        # self.options.add_argument("--headless") # Headless might trigger anti-bot or harder to debug
        self.options.add_argument("--mute-audio")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=800,600")
        
        # Initialize driver
        # simple init, assuming chromedriver is in path
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get("http://slither.io")
        
        # Wait for game to load and handle login
        time.sleep(3)
        self._handle_login()
        self.inject_override_script()

    def _handle_login(self):
        """
        Handles the initial login screen:
        1. Enter nickname in the text field
        2. Click the Play button
        """
        try:
            # Use JavaScript to enter nickname and start game
            # This is more reliable than Selenium element clicks for canvas-based games
            login_js = f"""
            // Try to find and fill the nickname field
            var nickField = document.querySelector('input[id="nick"]') || 
                           document.querySelector('input.nsi') ||
                           document.querySelector('input[placeholder*="nick"]');
            
            if (nickField) {{
                nickField.value = '{self.nickname}';
                nickField.dispatchEvent(new Event('input', {{ bubbles: true }}));
            }}
            
            // Try to click play button
            var playBtn = document.querySelector('.play-button') ||
                         document.querySelector('div.btnt') ||
                         document.querySelector('[onclick*="play"]');
                         
            if (playBtn) {{
                playBtn.click();
                return 'clicked';
            }}
            
            // Alternative: directly call play function if available
            if (typeof window.play === 'function') {{
                window.play();
                return 'played';
            }}
            
            // Another alternative: call connect
            if (typeof window.connect === 'function') {{
                window.connect();
                return 'connected';
            }}
            
            return 'no_button_found';
            """
            result = self.driver.execute_script(login_js)
            print(f"[LOGIN] Login attempt result: {result}")
            
            # Wait for game to actually start
            time.sleep(2)
            
        except Exception as e:
            print(f"[LOGIN] Error during login: {e}")
            # Try a simple approach - just wait and try connect()
            time.sleep(2)

    def inject_override_script(self):
        """
        Injects JS to:
        1. Disable high-end graphics (if possible)
        2. Hook mouse controls to allow virtual control via window.xm / window.ym
        3. Prepare the 'connect' hook for auto-restart
        """
        js_code = """
        // 1. Graphics Optimization
        if (window.want_quality) { window.want_quality = 0; }
        if (window.high_quality) { window.high_quality = false; }
        
        // 2. Control Hack
        // We overwrite the mouse move handler to stop it from updating xm/ym from cursor
        window.onmousemove = function(e) { 
            // Do nothing using actual mouse
            return; 
        };
        
        // Initialize our control variables if they don't exist (game usually has them)
        if (typeof window.xm === 'undefined') window.xm = 0;
        if (typeof window.ym === 'undefined') window.ym = 0;
        
        // Helper to force restart
        window.force_connect_game = function() {
            if (window.connect) {
                window.connect();
            } else {
                console.log("Connect function not found");
            }
        };

        console.log("SlitherBot: Overrides injected.");
        """
        self.driver.execute_script(js_code)

    def get_game_data(self):
        """
        Retrieves all necessary game state in a SINGLE JS call.
        Returns:
            dict with 'self', 'foods', 'enemies', 'dead'
        """
        # We need to carefully access game variables. 
        # Standard variables in Slither.io client usually include:
        # window.snake  -> The player's snake object
        # window.snakes -> All visible snakes
        # window.foods  -> All visible food
        # window.dead_mtm -> Death timestamp (if defined/not -1, usually means dead) or window.playing
        
        fetch_js = """
        function getGameState() {
            // Check death status
            // Depending on client version, 'snake' might be null if dead or not started
            var is_dead = false;
            if (!window.snake || (window.dead_mtm && window.dead_mtm !== -1)) {
                 is_dead = true;
            }

            if (is_dead) {
                return { dead: true };
            }

            // 1. My Snake Data
            var my_snake = {
                x: window.snake.xx, // xx is often used for interpolated x
                y: window.snake.yy,
                ang: window.snake.ang,
                sp: window.snake.sp, // speed
                len: window.snake.pts ? window.snake.pts.length : 0
            };

            // 2. Foods
            // window.foods is often a structured array or null
            var visible_foods = [];
            if (window.foods && window.foods.length) {
                for (var i = 0; i < window.foods.length; i++) {
                    var f = window.foods[i];
                    if (f && f.rx) { // rx/ry are render coordinates
                         visible_foods.push([f.rx, f.ry, f.sz || 1]);
                    }
                }
            }

            // 3. Enemies
            var visible_enemies = [];
            if (window.snakes && window.snakes.length) {
                for (var i = 0; i < window.snakes.length; i++) {
                    var s = window.snakes[i];
                    // Skip my own snake (window.snake has same id usually, or check object ref)
                    if (s === window.snake) continue; 
                    
                    // Collect body points
                    var pts = [];
                    if (s.pts) {
                        for (var j = 0; j < s.pts.length; j++) {
                            // Points structure might be simple objects {x, y} or optimized arrays
                            var p = s.pts[j];
                            if (p.x !== undefined) pts.push([p.x, p.y]);
                            else if (p.xx !== undefined) pts.push([p.xx, p.yy]);
                        }
                    }

                    visible_enemies.push({
                        id: s.id,
                        x: s.xx,
                        y: s.yy,
                        ang: s.ang,
                        sp: s.sp,
                        pts: pts
                    });
                }
            }

            return {
                dead: false,
                self: my_snake,
                foods: visible_foods,
                enemies: visible_enemies
            };
        }
        return getGameState();
        """
        try:
            return self.driver.execute_script(fetch_js)
        except Exception as e:
            # If something breaks (e.g. game reloading), assume dead or not ready
            print(f"Error fetching game data: {e}")
            return {"dead": True}

    def send_action(self, angle, boost):
        """
        Converts neural net output (angle, boost) to game controls.
        angle: float (0 to 1) mapped to 0..2PI or relative turn? 
               Usually NEAT outputs are specific. Let's assume angle is relative or absolute direction.
               The user requirement: "Convert angle to X/Y relative to snake head".
        """
        # Game control works by setting xm/ym to a point the snake should head towards.
        # We need current snake position to calculate vector.
        # But wait, to save a call, we can do calculation in JS or just estimate here 
        # based on last known 'self' data from get_game_data?
        # Actually proper way: The bot decides an absolute angle 0..2PI (or relative).
        # We calculate a target point far away in that direction.
        
        # angle comes from network. Let's assume it is 0..1 mapped to 0..2PI.
        target_angle = angle * 2 * math.pi
        
        # Distance to project the target point (arbitrary, just needs to be "far enough" to steer)
        r = 500 
        
        # We need the snake's head position to set xm/ym relative to screen center or map?
        # Slither.io window.xm/ym are usually coordinates relative to the *center of the screen* 
        # (where the snake head is locked), NOT absolute map coordinates.
        # Actually, in Slither, xm/ym are mouse coordinates relative to the window center (if properly hooked).
        # 0,0 is top left of screen usually? No, for game logic, it often wants the mouse position relative to center 
        # OR absolute map coordinates depending on implementation.
        #
        # Standard loop:
        # dx = mouseX - screenWidth/2
        # dy = mouseY - screenHeight/2
        # angle = atan2(dy, dx)
        #
        # So we can just set xm/ym to satisfy that angle.
        # We don't need the snake's absolute position if we control valid xm/ym which are essentially direction vectors 
        # from the center of the screen.
        
        # Let's verify standard Slither behavior: xm/ym are often absolute mouse coordinates on the canvas.
        # The game calculates direction = atan2(ym - snake.yy, xm - snake.xx).
        # SO we need to know snake.xx and snake.yy OR we need `get_game_data` to return them every frame 
        # (which it does).
        # However, to be robust within this method without passing state in, we can inject a script 
        # that uses the snake's CURRENT position directly in JS.
        
        control_js = f"""
        var ang = {target_angle};
        var boost = {1 if boost > 0.5 else 0};
        
        if (window.snake) {{
            // Calculate target point based on current head position
            var dist = 200;
            window.xm = window.snake.xx + Math.cos(ang) * dist;
            window.ym = window.snake.yy + Math.sin(ang) * dist;
            
            // Apply boost
            window.setAcceleration(boost); 
        }}
        """
        self.driver.execute_script(control_js)

    def force_restart(self):
        """
        Resets the game immediately.
        Re-injects control overrides after restart.
        """
        restart_js = """
        // Call connect directly - don't rely on our injected function
        if (typeof window.connect === 'function') {
            window.connect();
            return 'connected';
        } else {
            // Maybe we need to reload the page
            return 'no_connect';
        }
        """
        try:
            result = self.driver.execute_script(restart_js)
            if result == 'connected':
                # Wait a moment for the game to start
                time.sleep(1)
                # Re-inject our overrides (they get lost on game restart)
                self.inject_override_script()
        except Exception as e:
            print(f"Error in force_restart: {e}")
            # Try reloading the page as fallback
            self.driver.refresh()
            time.sleep(2)
            self.inject_override_script()

    def close(self):
        self.driver.quit()

