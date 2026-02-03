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

def log(msg):
    print(msg, flush=True)
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
        time.sleep(2)

    def _handle_login(self):
        """
        Handles the initial login screen:
        1. Enter nickname in the text field
        2. Click the Play button
        3. Close any popups/ads
        """
        try:
            # 0. Close extra windows (ads/popups)
            time.sleep(2) # Give ads a moment to pop up
            if len(self.driver.window_handles) > 1:
                log(f"[BROWSER] Closing {len(self.driver.window_handles)-1} extra tabs/ads.")
                main_handle = self.driver.current_window_handle
                for handle in self.driver.window_handles:
                    if handle != main_handle:
                        try:
                            self.driver.switch_to.window(handle)
                            self.driver.close()
                        except:
                            pass
                self.driver.switch_to.window(main_handle)

            log(f"[LOGIN] Attempting login with nickname: {self.nickname}")
            
            # 1. Fill Nickname via JS for speed and reliability
            self.driver.execute_script(f"document.getElementById('nick').value = '{self.nickname}';")
            
            # 2. Click Play Button (Verified selector: #playh .btnt)
            login_js = """
            var playBtn = document.querySelector('#playh .btnt') || 
                         document.querySelector('.btnt.btntg') ||
                         document.querySelector('#play-btn');
                         
            if (playBtn) {
                playBtn.click();
                return 'btn_clicked';
            }
            
            // Search by text fallback
            var divs = document.querySelectorAll('div, button, a');
            for (var i = 0; i < divs.length; i++) {
                if (divs[i].innerText && divs[i].innerText.trim() === 'Play') {
                    divs[i].click();
                    return 'text_btn_clicked';
                }
            }
            
            if (typeof window.connect === 'function') {
                window.connect();
                return 'connect_called';
            }
            return 'not_found';
            """
            result = self.driver.execute_script(login_js)
            log(f"[LOGIN] Play attempt result: {result}")
            
            # 3. Wait for game start
            start_wait = time.time()
            while time.time() - start_wait < 15:
                try:
                    # Check for alerts and close them
                    self.driver.switch_to.alert.accept()
                    log("[LOGIN] Alert accepted and closed.")
                except:
                    pass

                log(f"[LOGIN] Waiting for snake... ({int(time.time() - start_wait)}s)")
                is_playing = self.driver.execute_script("""
                    return (window.slither !== undefined && window.slither !== null) && 
                           (typeof window.dead_mtm === 'undefined' || window.dead_mtm === -1 || window.dead_mtm === null);
                """)
                if is_playing:
                    log("[LOGIN] Game started successfully!")
                    self.inject_override_script()
                    return True
                time.sleep(1)
            
            log("[LOGIN] Warning: Game didn't start. Saving diagnostic screenshot...")
            self.driver.save_screenshot("login_failure.png")
            return False
            
        except Exception as e:
            log(f"[LOGIN] Error: {e}")
            return False


    def inject_override_script(self):
        """
        Injects JS to:
        1. Disable high-end graphics
        2. Hook mouse controls to allow virtual control via window.xm / window.ym
        """
        js_code = """
        // 1. Graphics Optimization
        if (typeof window.want_quality !== 'undefined') window.want_quality = 0;
        if (typeof window.high_quality !== 'undefined') window.high_quality = false;
        
        // 2. Control Hack
        // Override onmousemove to prevent the game from updating xm/ym from real mouse
        window.onmousemove = function(e) {
            if (e && e.stopImmediatePropagation) e.stopImmediatePropagation();
            return false;
        };
        
        // 3. Initialize control variables
        if (typeof window.xm === 'undefined') window.xm = 0;
        if (typeof window.ym === 'undefined') window.ym = 0;
        
        console.log("SlitherBot: Overrides injected.");
        """
        try:
            self.driver.execute_script(js_code)
        except Exception as e:
            log(f"[OVERRIDE] Failed to inject: {e}")

    def get_game_data(self):
        """
        Retrieves all necessary game state in a SINGLE JS call.
        Uses updated variable names: slither, slithers, foods.
        """
        fetch_js = """
        function getGameState() {
            // State detection
            var playing = (window.slither !== undefined && window.slither !== null) && (typeof window.dead_mtm === 'undefined' || window.dead_mtm === -1 || window.dead_mtm === null);
            var in_menu = document.querySelector('#nick, #playh .btnt') !== null;
            
            if (!playing) {
                return { dead: true, in_menu: in_menu };
            }

            // 1. My Snake Data
            var my_snake = {
                x: window.slither.xx,
                y: window.slither.yy,
                ang: window.slither.ang,
                sp: window.slither.sp,
                len: window.slither.pts ? window.slither.pts.length : 0
            };

            // 2. Foods
            var visible_foods = [];
            if (window.foods && window.foods.length) {
                for (var i = 0; i < window.foods.length; i++) {
                    var f = window.foods[i];
                    if (f && f.rx) { 
                         visible_foods.push([f.rx, f.ry, f.sz || 1]);
                    }
                }
            }

            // 3. Enemies
            var visible_enemies = [];
            if (window.slithers && window.slithers.length) {
                for (var i = 0; i < window.slithers.length; i++) {
                    var s = window.slithers[i];
                    if (s === window.slither) continue; 
                    
                    var pts = [];
                    if (s.pts) {
                        for (var j = 0; j < s.pts.length; j++) {
                            var p = s.pts[j];
                            if (p.xx !== undefined) pts.push([p.xx, p.yy]);
                            else if (p.x !== undefined) pts.push([p.x, p.y]);
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
            result = self.driver.execute_script(fetch_js)
            if result is None:
                return {"dead": True}
            return result
        except Exception as e:
            return {"dead": True}

    def send_action(self, angle, boost):
        """
        Converts neural net output (angle, boost) to game controls.
        angle: 0 to 2PI (radians)
        boost: float/boolean (threshold > 0.5)
        """
        is_boost = 1 if boost > 0.5 else 0
        control_js = f"""
        var ang = {angle};
        var is_boost = {is_boost};
        
        if (window.slither) {{
            // Relative logic from elliott-wen
            // Center of screen is 0,0 for XM/YM when we override? 
            // Actually, game logic uses: var ang = Math.atan2(ym - h/2, xm - w/2);
            // So xm/ym are screen coordinates.
            // But elliott-wen uses: window.xm = cos(ang) * 300; window.ym = sin(ang) * 300;
            // This implies 0,0 is the center for the logic taking these inputs?
            // Wait, standard game loop calculates angle from center of screen (w/2, h/2).
            // If we override onmousemove, we might need to be careful.
            // But let's trust the reference implementation first.
            // Reference: window.xm = goalPos[0]; window.ym = goalPos[1];
            // goalPos = (cos(ang)*300, sin(ang)*300)
            
            var target_x = Math.cos(ang) * 300;
            var target_y = Math.sin(ang) * 300;
            
            window.xm = target_x;
            window.ym = target_y;
            
            if (typeof window.setAcceleration === 'function') {{
                window.setAcceleration(is_boost);
            }} else if (typeof window.setaccel === 'function') {{
                window.setaccel(is_boost);
            }}
        }}
        """
        try:
            self.driver.execute_script(control_js)
        except:
            pass

    def force_restart(self):
        """
        Resets the game immediately.
        Ensures we are back in playing state.
        """
        try:
            # Check if we are still alive
            is_playing = self.driver.execute_script("return (window.slither !== undefined && window.slither !== null) && (!window.dead_mtm || window.dead_mtm === -1)")
            
            if not is_playing:
                # If on main menu or dead, try standard login/connect
                log("[RESTART] Not playing. Attempting login/reconnect...")
                self._handle_login()
            else:
                # If alive, we force a connect to get a fresh start
                log("[RESTART] Forcing new connection...")
                self.driver.execute_script("if (window.connect) window.connect();")
            
            # Wait for game transition
            time.sleep(1)
            
            # RE-INJECT OVERRIDES (Crucial! Lost on many transitions)
            self.inject_override_script()
            
        except Exception as e:
            log(f"[RESTART] Error: {e}. Refreshing page...")
            self.driver.refresh()
            time.sleep(3)
            self._handle_login()
            self.inject_override_script()

    def close(self):
        self.driver.quit()

