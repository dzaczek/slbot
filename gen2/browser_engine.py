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
    MAX_FOODS = 300      # Max food items to process (Increased for better sensing)
    MAX_ENEMIES = 50     # Max enemy snakes to process (Increased to fix invisible snakes)
    MAX_BODY_PTS = 150   # Max body points per enemy (increased for better visibility)
    
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

    def scan_game_variables(self):
        """
        One-time scan of ALL game variables to find boundary-related ones.
        Returns a dict of potentially relevant variables.
        """
        scan_js = """
        var results = {};
        
        // 1. Scan ALL window-level variables for numbers in map-size range
        var numericVars = {};
        var arrayVars = {};
        var knownSkip = ['innerWidth','innerHeight','scrollX','scrollY','pageXOffset','pageYOffset',
                         'screenX','screenY','screenLeft','screenTop','outerWidth','outerHeight',
                         'devicePixelRatio','length','performance'];
        
        for (var key in window) {
            try {
                if (knownSkip.indexOf(key) >= 0) continue;
                var val = window[key];
                
                // Numeric variables (potential radius, center, size)
                if (typeof val === 'number' && !isNaN(val) && isFinite(val)) {
                    if (val > 100 && val < 200000) {
                        numericVars[key] = val;
                    }
                }
                
                // Arrays (potential boundary polygons)
                if (Array.isArray(val) && val.length > 3 && val.length < 10000) {
                    // Check if it contains numbers
                    if (typeof val[0] === 'number') {
                        arrayVars[key] = {length: val.length, first3: val.slice(0,3), last3: val.slice(-3)};
                    }
                }
            } catch(e) {}
        }
        
        results['numeric'] = numericVars;
        results['arrays'] = arrayVars;
        
        // 2. Specific slither.io variables to check
        var specific = {};
        var checkVars = ['grd','msx','msy','msc','bsr','bsc','bsc2','border','map_size',
                         'arena_size','game_radius','rfbx','rfby','cst','sector_size',
                         'grid_size','world_size','fmlts','fpsls','protocol_version',
                         'mcp','mcx','mcy','gla','glr','bmx','bmy','bmr'];
        
        for (var i = 0; i < checkVars.length; i++) {
            var v = checkVars[i];
            try {
                if (typeof window[v] !== 'undefined') {
                    var val = window[v];
                    if (typeof val === 'number') specific[v] = val;
                    else if (typeof val === 'string') specific[v] = val;
                    else if (Array.isArray(val)) specific[v] = 'Array(' + val.length + ')';
                    else if (typeof val === 'object' && val !== null) specific[v] = 'Object';
                    else specific[v] = typeof val;
                }
            } catch(e) {}
        }
        results['specific'] = specific;
        
        // 3. Snake position for reference
        if (window.slither) {
            results['snake_pos'] = {x: window.slither.xx, y: window.slither.yy};
        }
        
        // 4. Check for boundary drawing functions
        var funcNames = [];
        for (var key in window) {
            try {
                if (typeof window[key] === 'function') {
                    var src = window[key].toString().substring(0, 200);
                    if (src.indexOf('border') >= 0 || src.indexOf('bound') >= 0 || 
                        src.indexOf('pbx') >= 0 || src.indexOf('grd') >= 0 ||
                        src.indexOf('arena') >= 0 || src.indexOf('wall') >= 0) {
                        funcNames.push(key);
                    }
                }
            } catch(e) {}
        }
        results['boundary_funcs'] = funcNames;
        
        return results;
        """
        try:
            result = self.driver.execute_script(scan_js)
            return result
        except Exception as e:
            log(f"[SCAN] Failed: {e}")
            return None

    def get_game_data(self):
        """
        Retrieves game state in a SINGLE JS call.
        Includes limits on returned objects for performance.
        Also returns the actual view dimensions for correct scaling.
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

            // Get actual view dimensions from game
            // gsc = global scale, determines how much world space is visible
            var canvas = document.getElementById('mc') || document.querySelector('canvas');
            var canvasW = canvas ? canvas.width : 800;
            var canvasH = canvas ? canvas.height : 600;
            var gsc = window.gsc || 0.9;  // global scale (zoom level)
            
            // Calculate actual visible world area
            // Visible width in world units = canvas pixels / scale
            var viewWidth = canvasW / gsc;
            var viewHeight = canvasH / gsc;
            var viewRadius = Math.max(viewWidth, viewHeight) / 2;

            // My snake data
            var my_pts = [];
            if (window.slither.pts) {{
                var ptsLen = window.slither.pts.length;
                // Trim ghost tail dynamically (Increased)
                var trimCount = Math.floor(ptsLen * 0.15) + Math.floor((window.slither.sp || 5.7) * 2.0);
                var startIndex = Math.min(ptsLen - 1, trimCount);
                
                var step = Math.max(1, Math.floor(ptsLen / MAX_BODY_PTS));
                for (var j = startIndex; j < ptsLen && my_pts.length < MAX_BODY_PTS; j += step) {{
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

            // Foods (limited for performance) - only within view radius
            var visible_foods = [];
            if (window.foods && window.foods.length) {{
                var myX = my_snake.x;
                var myY = my_snake.y;
                var viewRadSq = viewRadius * viewRadius * 1.2; // slight buffer
                
                // Get closest foods first
                var foodList = [];
                for (var i = 0; i < window.foods.length && foodList.length < MAX_FOODS * 2; i++) {{
                    var f = window.foods[i];
                    if (f && f.rx) {{
                        var dx = f.rx - myX;
                        var dy = f.ry - myY;
                        var dist = dx*dx + dy*dy;
                        // Only include foods within view
                        if (dist < viewRadSq) {{
                            foodList.push([f.rx, f.ry, f.sz || 1, dist]);
                        }}
                    }}
                }}
                
                // Sort by distance and take closest
                foodList.sort(function(a, b) {{ return a[3] - b[3]; }});
                for (var i = 0; i < Math.min(foodList.length, MAX_FOODS); i++) {{
                    visible_foods.push([foodList[i][0], foodList[i][1], foodList[i][2]]);
                }}
            }}

            // Enemies - check if ANY part (head or body) is within view
            var visible_enemies = [];
            var totalSlithers = window.slithers ? window.slithers.length : 0;
            
            if (window.slithers && window.slithers.length) {{
                var myX = my_snake.x;
                var myY = my_snake.y;
                // Expanded search radius to prevent "invisible snakes" on minimap edge
                var searchRadSq = viewRadius * viewRadius * 25; // 5x radius for search (was 3x)
                var viewRadSq = viewRadius * viewRadius * 2.0;  // Expanded view for filtering points
                
                var enemyList = [];
                
                for (var i = 0; i < window.slithers.length; i++) {{
                    var s = window.slithers[i];
                    if (s === window.slither) continue;
                    if (!s || !s.pts) continue;
                    
                    // Check if head OR any body part is potentially visible or close enough to be relevant
                    var minDist = Infinity;
                    var hasVisiblePart = false;
                    
                    // Check head
                    var hdx = (s.xx || 0) - myX;
                    var hdy = (s.yy || 0) - myY;
                    var headDist = hdx*hdx + hdy*hdy;
                    if (headDist < viewRadSq) hasVisiblePart = true;
                    minDist = Math.min(minDist, headDist);
                    
                    // Check some body points (sample every 10th for speed)
                    if (!hasVisiblePart && s.pts && s.pts.length > 0) {{
                        var ptsLen = s.pts.length;
                        var trimCount = Math.floor(ptsLen * 0.1) + Math.floor((s.sp || 5.7) * 2.0);
                        var startIndex = Math.min(ptsLen - 1, trimCount);
                        
                        var step = Math.max(1, Math.floor(ptsLen / 20));
                        for (var j = startIndex; j < ptsLen; j += step) {{
                            var p = s.pts[j];
                            var px = p.xx !== undefined ? p.xx : (p.x || 0);
                            var py = p.yy !== undefined ? p.yy : (p.y || 0);
                            var bdx = px - myX;
                            var bdy = py - myY;
                            var bodyDist = bdx*bdx + bdy*bdy;
                            if (bodyDist < viewRadSq) {{
                                hasVisiblePart = true;
                                break;
                            }}
                            minDist = Math.min(minDist, bodyDist);
                        }}
                    }}
                    
                    // Include if any part is visible OR if close enough to matter
                    if (hasVisiblePart || minDist < searchRadSq) {{
                        enemyList.push([s, minDist, hasVisiblePart ? 0 : 1]);
                    }}
                }}
                
                // Sort: visible first, then by distance
                enemyList.sort(function(a, b) {{ 
                    if (a[2] !== b[2]) return a[2] - b[2];
                    return a[1] - b[1]; 
                }});
                
                // Take enemies (increased limit)
                for (var i = 0; i < Math.min(enemyList.length, MAX_ENEMIES); i++) {{
                    var s = enemyList[i][0];
                    
                    // Get body points - filter to only those in view
                    var pts = [];
                    if (s.pts) {{
                        var ptsLen = s.pts.length;
                        // Trim ghost tail dynamically (Increased)
                        var trimCount = Math.floor(ptsLen * 0.25) + Math.floor((s.sp || 5.7) * 4.0);
                        var startIndex = Math.min(ptsLen - 1, trimCount);
                        
                        var step = Math.max(1, Math.floor(ptsLen / MAX_BODY_PTS));
                        for (var j = startIndex; j < ptsLen && pts.length < MAX_BODY_PTS; j += step) {{
                            var p = s.pts[j];
                            var px = p.xx !== undefined ? p.xx : (p.x || 0);
                            var py = p.yy !== undefined ? p.yy : (p.y || 0);
                            
                            // Only include points that could be in view
                            var pdx = px - myX;
                            var pdy = py - myY;
                            if (pdx*pdx + pdy*pdy < viewRadSq * 2) {{
                                pts.push([px, py]);
                            }}
                        }}
                    }}

                    visible_enemies.push({{
                        id: s.id,
                        x: s.xx || 0,
                        y: s.yy || 0,
                        ang: s.ang || 0,
                        sp: s.sp || 0,
                        pts: pts
                    }});
                }}
            }}

            // DETECT MAP BOUNDARY
            var possibleMapVars = {{}};
            var boundarySource = 'none';
            var distToWall = 99999;
            var distFromCenter = 99999;
            
            var mapCenterX = 21600; // Default center
            var mapCenterY = 21600; // Default center
            var mapRadius = 21600;  // Default radius
            
            if (typeof window.grd !== 'undefined' && window.grd > 1000) {{
                mapCenterX = window.grd;
                mapCenterY = window.grd;
                mapRadius = window.grd * 0.98;
                boundarySource = 'grd';
                possibleMapVars['grd'] = window.grd;
            }} else {{
                boundarySource = 'default';
            }}

            distFromCenter = Math.sqrt(
                Math.pow(my_snake.x - mapCenterX, 2) +
                Math.pow(my_snake.y - mapCenterY, 2)
            );
            distToWall = mapRadius - distFromCenter;

            possibleMapVars['map_center'] = Math.round(mapCenterX) + ',' + Math.round(mapCenterY);
            possibleMapVars['map_radius'] = Math.round(mapRadius);
            possibleMapVars['dist_from_center'] = Math.round(distFromCenter);
            
            // Also capture pbx info for debugging (NOT used for wall detection)
            var pbxCount = 0;
            if (typeof window.pbx !== 'undefined' && window.pbx) {{
                for (var i = 0; i < window.pbx.length; i++) {{
                    if (window.pbx[i] !== 0 || (window.pby && window.pby[i] !== 0)) pbxCount++;
                }}
            }}
            
            possibleMapVars['pbx_count'] = pbxCount;
            possibleMapVars['FINAL_source'] = boundarySource;
            possibleMapVars['FINAL_dist_to_wall'] = Math.round(distToWall);
            
            return {{
                dead: false,
                self: my_snake,
                foods: visible_foods,
                enemies: visible_enemies,
                view_radius: viewRadius,
                gsc: gsc,
                dist_to_wall: distToWall,
                dist_from_center: distFromCenter,
                map_radius: mapRadius,
                map_center_x: mapCenterX,
                map_center_y: mapCenterY,
                debug: {{
                    total_slithers: totalSlithers,
                    visible_enemies: visible_enemies.length,
                    total_foods: window.foods ? window.foods.length : 0,
                    visible_foods: visible_foods.length,
                    dist_to_wall: Math.round(distToWall),
                    dist_from_center: Math.round(distFromCenter),
                    snake_x: Math.round(my_snake.x),
                    snake_y: Math.round(my_snake.y),
                    boundary_source: boundarySource,
                    map_vars: possibleMapVars
                }}
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

    def inject_view_plus_overlay(self):
        """
        Injects a visual overlay canvas that shows the bot's perception grid.
        Includes a CLIENT-SIDE RENDER LOOP for real-time (60fps) visualization without lag.
        """
        js_code = """
        (function() {
            // Remove existing overlay if present
            var existingContainer = document.getElementById('bot-vision-container');
            if (existingContainer) existingContainer.remove();
            
            // Create container for all elements
            var container = document.createElement('div');
            container.id = 'bot-vision-container';
            container.style.cssText = 'position:fixed;top:10px;right:10px;z-index:99999;pointer-events:none;';
            document.body.appendChild(container);
            
            // Create overlay canvas
            var canvas = document.createElement('canvas');
            canvas.id = 'bot-vision-overlay';
            canvas.width = 168; // 84 * 2
            canvas.height = 168;
            canvas.style.cssText = 'border:2px solid #00ff00;background:rgba(0,0,0,0.8);' +
                                   'border-radius:5px;display:block;margin-bottom:5px;';
            container.appendChild(canvas);
            
            // Create stats display
            var stats = document.createElement('div');
            stats.id = 'bot-vision-stats';
            stats.style.cssText = 'background:rgba(0,0,0,0.9);padding:6px 8px;border-radius:5px;' +
                                  'font-size:11px;color:#0f0;font-family:monospace;margin-top:5px;' +
                                  'line-height:1.4;pointer-events:none;';
            stats.innerHTML = 'Initializing Real-Time View...';
            container.appendChild(stats);
            
            // Create legend
            var legend = document.createElement('div');
            legend.id = 'bot-vision-legend';
            legend.style.cssText = 'background:rgba(0,0,0,0.8);padding:6px 8px;border-radius:5px;' +
                                   'font-size:10px;color:white;font-family:monospace;margin-top:5px;' +
                                   'display:flex;flex-wrap:wrap;gap:4px 10px;max-width:168px;';
            legend.innerHTML = '<span style="color:#00ff00">■ Food</span>' +
                               '<span style="color:#ff0000">■ Enemy</span>' +
                               '<span style="color:#00ffff">■ Self</span>' +
                               '<span style="color:#ff00ff">■ Wall</span>';
            container.appendChild(legend);

            // --- REAL-TIME RENDERER ---
            window.renderBotVisionLoop = function() {
                requestAnimationFrame(window.renderBotVisionLoop);

                var canvas = document.getElementById('bot-vision-overlay');
                if (!canvas) return;
                var ctx = canvas.getContext('2d');
                
                // Game checks
                if (!window.slither || !window.slither.xx) return;
                
                // 1. Calculate View Properties (Same logic as get_game_data)
                var gameCanvas = document.getElementById('mc') || document.querySelector('canvas');
                var canvasW = gameCanvas ? gameCanvas.width : 800;
                var canvasH = gameCanvas ? gameCanvas.height : 600;
                var gsc = window.gsc || 0.9;
                
                var viewWidth = canvasW / gsc;
                var viewHeight = canvasH / gsc;
                var viewRadius = Math.max(viewWidth, viewHeight) / 2;
                
                // Grid setup (84x84)
                var gridSize = 84;
                var scale = 2; // Pixel size on overlay
                
                // Clear and Status
                ctx.fillStyle = 'rgba(0,0,0,0.85)';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Debug Text Helper
                function drawDebugText(text, y) {
                    ctx.fillStyle = '#ffff00';
                    ctx.font = '10px monospace';
                    ctx.textAlign = 'left';
                    ctx.fillText(text, 5, y);
                }

                // Helper: World to Matrix
                var myX = window.slither.xx;
                var myY = window.slither.yy;
                var matrixScale = gridSize / (viewRadius * 2);
                
                // ... (rest of toGrid) ...
                
                function toGrid(wx, wy) {
                    var dx = wx - myX;
                    var dy = wy - myY;
                    var gx = (dx * matrixScale) + (gridSize / 2);
                    var gy = (dy * matrixScale) + (gridSize / 2);
                    return {x: gx, y: gy};
                }

                // 2. Draw Food (Green)
                ctx.fillStyle = '#00ff00';
                if (window.foods) {
                    var viewRadSq = viewRadius * viewRadius * 1.2;
                    for (var i = 0; i < window.foods.length; i++) {
                        var f = window.foods[i];
                        if (!f || !f.rx) continue;
                        
                        // Visibility check
                        var dx = f.rx - myX;
                        var dy = f.ry - myY;
                        if (dx*dx + dy*dy > viewRadSq) continue;
                        
                        var p = toGrid(f.rx, f.ry);
                        if (p.x >= 0 && p.x < gridSize && p.y >= 0 && p.y < gridSize) {
                            var sz = (f.sz || 1) * matrixScale * scale * 2.5; // Reduced scaling (was 10)
                            ctx.beginPath();
                            ctx.arc(p.x * scale, p.y * scale, Math.max(1.5, sz), 0, 2*Math.PI);
                            ctx.fill();
                        }
                    }
                }

                // 3. Draw Enemies (Red)
                if (window.slithers) {
                    var viewRadSq = viewRadius * viewRadius * 1.5;
                    for (var i = 0; i < window.slithers.length; i++) {
                        var s = window.slithers[i];
                        if (s === window.slither) continue;
                        
                        // Calculate REAL thickness
                        // Base width is ~29. sc is scale.
                        var snakeWidth = (s.sc || 1) * 29;
                        var gridWidth = snakeWidth * matrixScale * scale;
                        var drawRadius = Math.max(scale, gridWidth / 2);

                        // Body
                        ctx.fillStyle = '#ff6600'; // Body color
                        if (s.pts) {
                             var ptsLen = s.pts.length;
                             // TRIM TAIL DYNAMICALLY (Increased):
                             // Remove ghost points from tail (start of array)
                             // Trimming ~25% of history points + speed offset to match real visual length
                             var trimCount = Math.floor(ptsLen * 0.25) + Math.floor((s.sp || 5.7) * 4.0);
                             var startIndex = Math.min(ptsLen - 1, trimCount);
                             
                             var step = Math.max(1, Math.floor(ptsLen / 100));
                             
                             for (var j = startIndex; j < ptsLen; j+=step) {
                                 var pt = s.pts[j];
                                 var px = pt.xx !== undefined ? pt.xx : (pt.x || 0);
                                 var py = pt.yy !== undefined ? pt.yy : (pt.y || 0);
                                 
                                 var pdx = px - myX;
                                 var pdy = py - myY;
                                 if (pdx*pdx + pdy*pdy > viewRadSq) continue;

                                 var pp = toGrid(px, py);
                                 if (pp.x >= -5 && pp.x < gridSize+5 && pp.y >= -5 && pp.y < gridSize+5) {
                                     // Tapering logic: Tail is at index 0
                                     // Reduce radius for the first 20% of the body
                                     var taperFactor = 1.0;
                                     if (j < ptsLen * 0.2) {
                                         taperFactor = 0.3 + (0.7 * (j / (ptsLen * 0.2)));
                                     }
                                     
                                     var currentRadius = drawRadius * taperFactor;
                                     
                                     ctx.beginPath();
                                     ctx.arc(pp.x * scale, pp.y * scale, currentRadius, 0, 2*Math.PI);
                                     ctx.fill();
                                 }
                             }
                        }

                        // Head
                        var hx = s.xx || 0;
                        var hy = s.yy || 0;
                        var hp = toGrid(hx, hy);
                        if (hp.x >= -5 && hp.x < gridSize+5 && hp.y >= -5 && hp.y < gridSize+5) {
                            ctx.fillStyle = '#ff0000'; // Head color
                            ctx.beginPath();
                            ctx.arc(hp.x * scale, hp.y * scale, drawRadius * 1.2, 0, 2*Math.PI);
                            ctx.fill();
                        }
                    }
                }

                // 4. Draw Self (Cyan/Blue)
                var myWidth = (window.slither.sc || 1) * 29;
                var myGridWidth = myWidth * matrixScale * scale;
                var myRadius = Math.max(scale, myGridWidth / 2);

                // Body
                ctx.fillStyle = '#0088ff';
                if (window.slither.pts) {
                    var ptsLen = window.slither.pts.length;
                    // TRIM TAIL DYNAMICALLY (Self - Increased):
                    var trimCount = Math.floor(ptsLen * 0.15) + Math.floor((window.slither.sp || 5.7) * 2.0);
                    var startIndex = Math.min(ptsLen - 1, trimCount);
                    
                    var step = Math.max(1, Math.floor(ptsLen / 100));
                    
                    for (var j = startIndex; j < ptsLen; j+=step) {
                        var pt = window.slither.pts[j];
                        var px = pt.xx !== undefined ? pt.xx : (pt.x || 0);
                        var py = pt.yy !== undefined ? pt.yy : (pt.y || 0);
                        var pp = toGrid(px, py);
                        
                        // Tapering logic for self
                        var taperFactor = 1.0;
                        if (j < ptsLen * 0.2) {
                            taperFactor = 0.3 + (0.7 * (j / (ptsLen * 0.2)));
                        }
                        var currentRadius = myRadius * taperFactor;
                        
                        ctx.beginPath();
                        ctx.arc(pp.x * scale, pp.y * scale, currentRadius, 0, 2*Math.PI);
                        ctx.fill();
                    }
                }
                // Head
                ctx.fillStyle = '#00ffff';
                var sp = toGrid(myX, myY);
                ctx.beginPath();
                ctx.arc(sp.x * scale, sp.y * scale, myRadius * 1.2, 0, 2*Math.PI);
                ctx.fill();

                // 5. Draw Wall/Boundary (Magenta)
                // DETECT DYNAMIC MAP BOUNDARY
            // Center = (grd, grd), Radius = grd * 0.98
            var mapRadius = 21600;
            var mapCenterX = 0;
            var mapCenterY = 0;
            var boundarySource = 'def';
            
            if (typeof window.grd !== 'undefined' && window.grd > 1000) {
                mapCenterX = window.grd;
                mapCenterY = window.grd;
                mapRadius = window.grd * 0.98;  // Confirmed by reference bots
                boundarySource = 'grd';
            } else {
                mapCenterX = 21600;
                mapCenterY = 21600;
                mapRadius = 21600;
                boundarySource = 'def';
            }

                // Scan grid for wall pixels (expensive but accurate visualization)
                // Optimization: Only scan if near wall
                var distFromCenter = Math.sqrt(Math.pow(myX-mapCenterX, 2) + Math.pow(myY-mapCenterY, 2));
                var distToWall = mapRadius - distFromCenter;
                
                // Debug values for stats (store in window for update function to read)
                window._botVisionDebug = {
                    mapRadius: Math.round(mapRadius),
                    distToWall: Math.round(distToWall),
                    boundarySource: boundarySource,
                    viewRadius: Math.round(viewRadius),
                    distFromCenter: Math.round(distFromCenter)
                };
                
                // DRAW WALL - Only when within 5000 world units
                if (distToWall < 5000) {
                    var centerGrid = toGrid(mapCenterX, mapCenterY);
                    var wallCx = centerGrid.x * scale;
                    var wallCy = centerGrid.y * scale;
                    var r = mapRadius * matrixScale * scale;

                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(-100, -100, canvas.width+200, canvas.height+200);
                    ctx.arc(wallCx, wallCy, r, 0, 2 * Math.PI, true);
                    ctx.fillStyle = 'rgba(255, 0, 255, 0.4)'; 
                    ctx.fill('evenodd');
                    
                    ctx.strokeStyle = '#ff00ff';
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.arc(wallCx, wallCy, r, 0, 2 * Math.PI);
                    ctx.stroke();
                    ctx.restore();
                }

                // RADAR: Show direction and distance to wall
                if (mapRadius > 0) { 
                    var vcx = myX - mapCenterX;
                    var vcy = myY - mapCenterY;
                    var distFromMapCenter = Math.sqrt(vcx*vcx + vcy*vcy);
                    
                    var nx = 0, ny = -1;
                    if (distFromMapCenter > 1) {
                         nx = vcx / distFromMapCenter;
                         ny = vcy / distFromMapCenter;
                    }
                    
                    // Color based on danger level
                    var radarColor = '#ff00ff';
                    var radarAlpha = 0.4;
                    if (distToWall < 1000) {
                        radarColor = '#ff0000';  // RED when very close
                        radarAlpha = 1.0;
                    } else if (distToWall < 3000) {
                        radarColor = '#ff00ff';  // Magenta when medium
                        radarAlpha = 0.8;
                    } else {
                        radarAlpha = 0.3;  // Dim when far
                    }
                    
                    var centerP = {x: gridSize/2, y: gridSize/2};
                    var arrowDist = (gridSize/2 - 8) * scale;
                    var ax = centerP.x * scale + nx * arrowDist;
                    var ay = centerP.y * scale + ny * arrowDist;
                    
                    ctx.globalAlpha = radarAlpha;
                    
                    // Draw Arrow dot
                    ctx.beginPath();
                    ctx.arc(ax, ay, 5, 0, 2*Math.PI);
                    ctx.fillStyle = radarColor;
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                    
                    // Arrow Tip
                    ctx.beginPath();
                    ctx.moveTo(ax + nx*8, ay + ny*8);
                    ctx.lineTo(ax - ny*4, ay + nx*4);
                    ctx.lineTo(ax + ny*4, ay - nx*4);
                    ctx.fillStyle = 'white';
                    ctx.fill();
                    
                    // Distance text
                    ctx.fillStyle = 'white';
                    ctx.font = 'bold 12px monospace';
                    ctx.textAlign = 'center';
                    ctx.shadowColor = "black";
                    ctx.shadowBlur = 2;
                    ctx.fillText(Math.round(distToWall), ax - nx*20, ay - ny*20);
                    ctx.shadowBlur = 0;
                    ctx.globalAlpha = 1.0;
                }

                // Grid lines
                ctx.strokeStyle = 'rgba(255,255,255,0.1)';
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                for (var i=0; i<=gridSize; i+=10) {
                    ctx.moveTo(i*scale, 0); ctx.lineTo(i*scale, gridSize*scale);
                    ctx.moveTo(0, i*scale); ctx.lineTo(gridSize*scale, i*scale);
                }
                ctx.stroke();

                // Crosshair
                var cx = (gridSize/2)*scale + scale/2;
                var cy = (gridSize/2)*scale + scale/2;
                ctx.strokeStyle = 'white';
                ctx.beginPath();
                ctx.moveTo(cx-5, cy); ctx.lineTo(cx+5, cy);
                ctx.moveTo(cx, cy-5); ctx.lineTo(cx, cy+5);
                ctx.stroke();
            };
            
            // Start loop
            console.log('SlitherBot: Starting Real-Time Render Loop');
            window.renderBotVisionLoop();
        })();
        """
        try:
            self.driver.execute_script(js_code)
            log("[VIEW-PLUS] Real-Time Overlay injected")
        except Exception as e:
            log(f"[VIEW-PLUS] Failed to inject overlay: {e}")

    def update_view_plus_overlay(self, matrix=None, gsc=None, view_radius=None, debug_info=None):
        """
        Updates ONLY the stats text. The grid is now rendered Client-Side (JS) for 60fps performance.
        Matrix argument is ignored but kept for compatibility.
        """
        # Stats values
        gsc_val = float(gsc) if gsc else 0.0
        
        # Debug info defaults
        enemies_total = 0
        enemies_visible = 0
        foods_visible = 0
        dist_to_wall = 0
        map_radius = 21600
        snake_x = 0
        snake_y = 0
        boundary_source = '?'
        
        if debug_info:
            enemies_total = debug_info.get('total_slithers', 0)
            enemies_visible = debug_info.get('visible_enemies', 0)
            foods_visible = debug_info.get('visible_foods', 0)
            dist_to_wall = debug_info.get('dist_to_wall', 0)
            map_radius = debug_info.get('map_radius', 21600)
            snake_x = debug_info.get('snake_x', 0)
            snake_y = debug_info.get('snake_y', 0)
            boundary_source = debug_info.get('boundary_source', '?')

        js_code = f"""
        (function() {{
            var stats = document.getElementById('bot-vision-stats');
            if (stats) {{
                // Use Client-Side Debug Data if available (more accurate to what's rendered)
                var d = window._botVisionDebug;
                if (d) {{
                     var wallStatus = '';
                     if (d.viewRadius && d.distToWall < d.viewRadius * 2.5) wallStatus = ' <span style="color:#ff00ff;font-weight:bold">⚠ WALL</span>';
                     
                     stats.innerHTML = 'src:' + d.boundarySource + ' (' + (d.pbxCount||0) + ' pts) R:' + d.mapRadius + 
                                       '<br>pos:{snake_x},{snake_y}<br>wall:' + d.distToWall + wallStatus +
                                       '<br>en:{enemies_visible}/{enemies_total} fd:{foods_visible}';
                }} else {{
                     stats.innerHTML = 'src:{boundary_source} R:{map_radius}<br>pos:{snake_x},{snake_y}<br>wall:{dist_to_wall}<br>en:{enemies_visible}/{enemies_total} fd:{foods_visible}';
                }}
            }}
        }})();
        """
        try:
            self.driver.execute_script(js_code)
        except:
            pass

