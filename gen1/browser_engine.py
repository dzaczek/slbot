"""
Browser Engine for Slither.io Bot
Manages Chrome/Chromium instances and game communication.
"""

import time
import math
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

def log(msg):
    print(msg, flush=True)


class SlitherBrowser:
    """
    Manages browser instance using Selenium.
    Handles JS bridge to communicate with Slither.io client.
    """
    
    # Limits for performance
    MAX_FOODS = 100      # Max food items to process
    MAX_ENEMIES = 15     # Max enemy snakes to process
    MAX_BODY_PTS = 100   # Max body points per enemy (increased for better visibility)
    
    def __init__(self, headless=True, nickname="NEATBot"):
        self.nickname = nickname
        self.options = Options()
        
        if headless:
            self.options.add_argument("--headless=new")
            
        self.options.add_argument("--mute-audio")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--disable-dev-shm-usage")
        self.options.add_argument("--window-size=800,600")
        self.options.add_argument("--disable-extensions")
        self.options.add_argument("--disable-infobars")
        self.options.add_argument("--disable-notifications")
        
        # Performance optimizations
        self.options.add_argument("--disable-software-rasterizer")
        self.options.add_argument("--disable-background-networking")
        self.options.add_argument("--disable-sync")
        self.options.add_argument("--disable-translate")
        self.options.add_argument("--metrics-recording-only")
        self.options.add_argument("--no-first-run")
        
        # Disable images for faster loading (optional - might affect gameplay detection)
        prefs = {
            "profile.managed_default_content_settings.images": 2,
            "profile.default_content_setting_values.notifications": 2
        }
        self.options.add_experimental_option("prefs", prefs)
        
        # Initialize driver
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.set_page_load_timeout(30)
        self.driver.get("http://slither.io")
        time.sleep(3)

    def _handle_login(self):
        """
        Handles the initial login screen.
        """
        try:
            # Close extra windows (ads/popups)
            time.sleep(1)
            if len(self.driver.window_handles) > 1:
                log(f"[BROWSER] Closing {len(self.driver.window_handles)-1} extra tabs.")
                main_handle = self.driver.current_window_handle
                for handle in self.driver.window_handles:
                    if handle != main_handle:
                        try:
                            self.driver.switch_to.window(handle)
                            self.driver.close()
                        except:
                            pass
                self.driver.switch_to.window(main_handle)

            log(f"[LOGIN] Attempting login: {self.nickname}")
            
            # Fill nickname and click play
            self.driver.execute_script(f"document.getElementById('nick').value = '{self.nickname}';")
            
            login_js = """
            var playBtn = document.querySelector('#playh .btnt') || 
                         document.querySelector('.btnt.btntg') ||
                         document.querySelector('#play-btn');
                         
            if (playBtn) {
                playBtn.click();
                return 'btn_clicked';
            }
            
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
            log(f"[LOGIN] Play attempt: {result}")
            
            # Wait for game start
            start_wait = time.time()
            while time.time() - start_wait < 15:
                try:
                    self.driver.switch_to.alert.accept()
                except:
                    pass

                is_playing = self.driver.execute_script("""
                    return (window.slither !== undefined && window.slither !== null) && 
                           (typeof window.dead_mtm === 'undefined' || window.dead_mtm === -1 || window.dead_mtm === null);
                """)
                if is_playing:
                    log("[LOGIN] Game started!")
                    self.inject_override_script()
                    return True
                time.sleep(0.5)
            
            log("[LOGIN] Warning: Game didn't start in time.")
            return False
            
        except Exception as e:
            log(f"[LOGIN] Error: {e}")
            return False

    def inject_override_script(self):
        """
        Injects JS overrides for bot control.
        """
        js_code = """
        // Graphics optimization
        if (typeof window.want_quality !== 'undefined') window.want_quality = 0;
        if (typeof window.high_quality !== 'undefined') window.high_quality = false;
        if (typeof window.render_mode !== 'undefined') window.render_mode = 1;
        
        // Disable visual effects
        window.redraw = window.redraw || function(){};
        
        // Override mouse controls
        window.onmousemove = function(e) {
            if (e && e.stopImmediatePropagation) e.stopImmediatePropagation();
            return false;
        };
        
        // Initialize control variables
        if (typeof window.xm === 'undefined') window.xm = 0;
        if (typeof window.ym === 'undefined') window.ym = 0;
        
        console.log("SlitherBot: Controls injected.");
        """
        try:
            self.driver.execute_script(js_code)
        except Exception as e:
            log(f"[OVERRIDE] Failed: {e}")

    def get_game_data(self):
        """
        Retrieves game state in a SINGLE JS call.
        Includes limits on returned objects for performance.
        """
        fetch_js = f"""
        function getGameState() {{
            var MAX_FOODS = {self.MAX_FOODS};
            var MAX_ENEMIES = {self.MAX_ENEMIES};
            var MAX_BODY_PTS = {self.MAX_BODY_PTS};
            
            var playing = (window.slither !== undefined && window.slither !== null) && 
                         (typeof window.dead_mtm === 'undefined' || window.dead_mtm === -1 || window.dead_mtm === null);
            var in_menu = document.querySelector('#nick, #playh .btnt') !== null;
            
            if (!playing) {{
                return {{ dead: true, in_menu: in_menu }};
            }}

            // My snake data
            var my_pts = [];
            if (window.slither.pts) {{
                var step = Math.max(1, Math.floor(window.slither.pts.length / MAX_BODY_PTS));
                for (var j = 0; j < window.slither.pts.length && my_pts.length < MAX_BODY_PTS; j += step) {{
                    var p = window.slither.pts[j];
                    if (p.xx !== undefined) my_pts.push([p.xx, p.yy]);
                    else if (p.x !== undefined) my_pts.push([p.x, p.y]);
                }}
            }}

            var my_snake = {{
                x: window.slither.xx,
                y: window.slither.yy,
                ang: window.slither.ang,
                sp: window.slither.sp,
                len: window.slither.pts ? window.slither.pts.length : 0,
                pts: my_pts
            }};

            // Foods (limited for performance)
            var visible_foods = [];
            if (window.foods && window.foods.length) {{
                var myX = my_snake.x;
                var myY = my_snake.y;
                
                // Get closest foods first
                var foodList = [];
                for (var i = 0; i < window.foods.length && foodList.length < MAX_FOODS * 2; i++) {{
                    var f = window.foods[i];
                    if (f && f.rx) {{
                        var dx = f.rx - myX;
                        var dy = f.ry - myY;
                        var dist = dx*dx + dy*dy;
                        foodList.push([f.rx, f.ry, f.sz || 1, dist]);
                    }}
                }}
                
                // Sort by distance and take closest
                foodList.sort(function(a, b) {{ return a[3] - b[3]; }});
                for (var i = 0; i < Math.min(foodList.length, MAX_FOODS); i++) {{
                    visible_foods.push([foodList[i][0], foodList[i][1], foodList[i][2]]);
                }}
            }}

            // Enemies (limited for performance)
            var visible_enemies = [];
            if (window.slithers && window.slithers.length) {{
                var enemyList = [];
                var myX = my_snake.x;
                var myY = my_snake.y;
                
                for (var i = 0; i < window.slithers.length; i++) {{
                    var s = window.slithers[i];
                    if (s === window.slither) continue;
                    
                    var dx = s.xx - myX;
                    var dy = s.yy - myY;
                    var dist = dx*dx + dy*dy;
                    enemyList.push([s, dist]);
                }}
                
                // Sort by distance
                enemyList.sort(function(a, b) {{ return a[1] - b[1]; }});
                
                // Take closest enemies
                for (var i = 0; i < Math.min(enemyList.length, MAX_ENEMIES); i++) {{
                    var s = enemyList[i][0];
                    
                    // Get limited body points
                    var pts = [];
                    if (s.pts) {{
                        var step = Math.max(1, Math.floor(s.pts.length / MAX_BODY_PTS));
                        for (var j = 0; j < s.pts.length && pts.length < MAX_BODY_PTS; j += step) {{
                            var p = s.pts[j];
                            if (p.xx !== undefined) pts.push([p.xx, p.yy]);
                            else if (p.x !== undefined) pts.push([p.x, p.y]);
                        }}
                    }}

                    visible_enemies.push({{
                        id: s.id,
                        x: s.xx,
                        y: s.yy,
                        ang: s.ang,
                        sp: s.sp,
                        pts: pts
                    }});
                }}
            }}

            return {{
                dead: false,
                self: my_snake,
                foods: visible_foods,
                enemies: visible_enemies
            }};
        }}
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
        Sends steering commands to the game.
        angle: target direction in radians (-PI to PI)
        boost: 0 or 1
        """
        is_boost = 1 if boost > 0.5 else 0
        
        control_js = f"""
        var ang = {angle};
        var is_boost = {is_boost};
        
        if (window.slither) {{
            var canvas = document.getElementById('mc') || document.querySelector('canvas');
            var w = canvas ? canvas.width : 800;
            var h = canvas ? canvas.height : 600;
            var centerX = w / 2;
            var centerY = h / 2;
            
            var radius = 300;
            var target_x = centerX + Math.cos(ang) * radius;
            var target_y = centerY + Math.sin(ang) * radius;
            
            window.xm = target_x;
            window.ym = target_y;
            
            // Boost control
            if (is_boost) {{
                window.accelerating = true;
                if (window.bso2) {{
                    try {{ window.bso2.send(new Uint8Array([253])); }} catch(e) {{}}
                }}
            }} else {{
                window.accelerating = false;
            }}
        }}
        """
        try:
            self.driver.execute_script(control_js)
        except:
            pass

    def force_restart(self):
        """
        Resets the game state.
        """
        try:
            is_playing = self.driver.execute_script("""
                return (window.slither !== undefined && window.slither !== null) && 
                       (!window.dead_mtm || window.dead_mtm === -1)
            """)
            
            if not is_playing:
                log("[RESTART] Not playing. Reconnecting...")
                self._handle_login()
            else:
                log("[RESTART] Forcing new connection...")
                self.driver.execute_script("if (window.connect) window.connect();")
            
            time.sleep(0.5)
            self.inject_override_script()
            
        except Exception as e:
            log(f"[RESTART] Error: {e}. Refreshing page...")
            self.driver.refresh()
            time.sleep(3)
            self._handle_login()
            self.inject_override_script()

    def close(self):
        """Close browser instance."""
        try:
            self.driver.quit()
        except:
            pass
