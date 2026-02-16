#!/usr/bin/env python3
"""Quick diagnostic: which steering method actually changes wang/ehang?"""
import time, math, sys
sys.path.insert(0, '.')
from browser_engine import SlitherBrowser

def read_angles(browser):
    return browser.driver.execute_script("""
        if (!window.slither) return null;
        return {
            ang: window.slither.ang,
            wang: window.slither.wang,
            ehang: window.slither.ehang
        };
    """)

def wait_alive(browser, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        d = browser.get_game_data()
        if d and not d.get('dead', True):
            return True
        time.sleep(0.5)
    return False

def deg(rad):
    return f"{math.degrees(rad):+.1f}°"

import os
os.makedirs("logs", exist_ok=True)
_logf = open("logs/steer_diag.log", "w")
_orig_print = print
def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs['file'] = _logf
    kwargs['flush'] = True
    _orig_print(*args, **kwargs)

print("=== STEERING DIAGNOSTIC ===\n")

browser = SlitherBrowser(headless=False, nickname="DiagTest")
browser._handle_login()
if not wait_alive(browser):
    print("FAILED: snake never spawned")
    browser.driver.quit()
    exit(1)

time.sleep(1)

# Canvas info
canvas_info = browser.driver.execute_script("""
    var c = document.getElementById('mc') || document.querySelector('canvas');
    if (!c) return null;
    var r = c.getBoundingClientRect();
    return {left: r.left, top: r.top, width: r.width, height: r.height,
            cx: r.left + r.width/2, cy: r.top + r.height/2,
            cw: c.width, ch: c.height};
""")
print(f"Canvas: {canvas_info}\n")

# Initial state
a0 = read_angles(browser)
print(f"INITIAL: ang={deg(a0['ang'])}  wang={deg(a0['wang'])}  ehang={deg(a0['ehang'])}\n")

# Target: 90° CW from current heading
target = a0['ang'] + 1.57  # +90°

# ─────────────────────────────────────────────────────────────
# METHOD 1: CDP Input.dispatchMouseEvent
# ─────────────────────────────────────────────────────────────
print("METHOD 1: CDP Input.dispatchMouseEvent")
cx, cy = canvas_info['cx'], canvas_info['cy']
mx = cx + math.cos(target) * 300
my = cy + math.sin(target) * 300
print(f"  Sending mouseMoved to ({mx:.0f}, {my:.0f})")
try:
    browser.driver.execute_cdp_cmd('Input.dispatchMouseEvent', {
        'type': 'mouseMoved',
        'x': mx,
        'y': my,
    })
    time.sleep(0.5)
    a1 = read_angles(browser)
    wang_changed = abs(a1['wang'] - a0['wang']) > 0.05
    print(f"  RESULT: wang={deg(a1['wang'])}  ehang={deg(a1['ehang'])}  "
          f"wang_changed={wang_changed}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# METHOD 2: Synthetic MouseEvent on canvas (not trusted)
# ─────────────────────────────────────────────────────────────
print("\nMETHOD 2: Synthetic MouseEvent on canvas")
try:
    browser.driver.execute_script(f"""
        var c = document.getElementById('mc') || document.querySelector('canvas');
        var r = c.getBoundingClientRect();
        var cx = r.left + r.width/2;
        var cy = r.top + r.height/2;
        var mx = cx + Math.cos({target}) * 300;
        var my = cy + Math.sin({target}) * 300;
        var evt = new MouseEvent('mousemove', {{
            clientX: mx, clientY: my, bubbles: true, cancelable: true
        }});
        c.dispatchEvent(evt);
    """)
    time.sleep(0.5)
    a2 = read_angles(browser)
    wang_changed = abs(a2['wang'] - a0['wang']) > 0.05
    print(f"  RESULT: wang={deg(a2['wang'])}  ehang={deg(a2['ehang'])}  "
          f"wang_changed={wang_changed}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# METHOD 3: Synthetic MouseEvent on document
# ─────────────────────────────────────────────────────────────
print("\nMETHOD 3: Synthetic MouseEvent on document")
try:
    browser.driver.execute_script(f"""
        var c = document.getElementById('mc') || document.querySelector('canvas');
        var r = c.getBoundingClientRect();
        var cx = r.left + r.width/2;
        var cy = r.top + r.height/2;
        var mx = cx + Math.cos({target}) * 300;
        var my = cy + Math.sin({target}) * 300;
        var evt = new MouseEvent('mousemove', {{
            clientX: mx, clientY: my, bubbles: true, cancelable: true
        }});
        document.dispatchEvent(evt);
    """)
    time.sleep(0.5)
    a3 = read_angles(browser)
    wang_changed = abs(a3['wang'] - a0['wang']) > 0.05
    print(f"  RESULT: wang={deg(a3['wang'])}  ehang={deg(a3['ehang'])}  "
          f"wang_changed={wang_changed}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# METHOD 4: Synthetic MouseEvent on window
# ─────────────────────────────────────────────────────────────
print("\nMETHOD 4: Synthetic MouseEvent on window")
try:
    browser.driver.execute_script(f"""
        var c = document.getElementById('mc') || document.querySelector('canvas');
        var r = c.getBoundingClientRect();
        var cx = r.left + r.width/2;
        var cy = r.top + r.height/2;
        var mx = cx + Math.cos({target}) * 300;
        var my = cy + Math.sin({target}) * 300;
        var evt = new MouseEvent('mousemove', {{
            clientX: mx, clientY: my, bubbles: true, cancelable: true
        }});
        window.dispatchEvent(evt);
    """)
    time.sleep(0.5)
    a4 = read_angles(browser)
    wang_changed = abs(a4['wang'] - a0['wang']) > 0.05
    print(f"  RESULT: wang={deg(a4['wang'])}  ehang={deg(a4['ehang'])}  "
          f"wang_changed={wang_changed}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# METHOD 5: Direct wang + ehang override
# ─────────────────────────────────────────────────────────────
print("\nMETHOD 5: Direct wang + ehang override via JS")
try:
    browser.driver.execute_script(f"""
        window.slither.wang = {target};
        window.slither.ehang = {target};
    """)
    time.sleep(0.5)
    a5 = read_angles(browser)
    wang_changed = abs(a5['wang'] - a0['wang']) > 0.05
    ehang_changed = abs(a5['ehang'] - a0['ehang']) > 0.05
    print(f"  RESULT: wang={deg(a5['wang'])}  ehang={deg(a5['ehang'])}  "
          f"ang={deg(a5['ang'])}  wang_changed={wang_changed}  ehang_changed={ehang_changed}")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────
# METHOD 6: Find mouse handler — check where game listens
# ─────────────────────────────────────────────────────────────
print("\nMETHOD 6: Detect mouse handler location")
try:
    handler_info = browser.driver.execute_script("""
        var result = {};
        result.docOnMouseMove = typeof document.onmousemove;
        result.winOnMouseMove = typeof window.onmousemove;
        var c = document.getElementById('mc') || document.querySelector('canvas');
        result.canvasOnMouseMove = c ? typeof c.onmousemove : 'no canvas';

        // Check for global xm/ym variables
        result.hasXm = typeof window.xm !== 'undefined';
        result.hasYm = typeof window.ym !== 'undefined';
        result.xm = window.xm;
        result.ym = window.ym;

        // Check for global mouse vars with other names
        result.hasMx = typeof window.mx !== 'undefined';
        result.hasMy = typeof window.my !== 'undefined';
        result.hasLmx = typeof window.lmx !== 'undefined';
        result.hasLmy = typeof window.lmy !== 'undefined';
        result.lmx = window.lmx;
        result.lmy = window.lmy;

        // Try to find the handler by checking common patterns
        if (c && typeof c.onmousemove === 'function') {
            result.canvasHandlerSrc = c.onmousemove.toString().substring(0, 200);
        }
        if (typeof document.onmousemove === 'function') {
            result.docHandlerSrc = document.onmousemove.toString().substring(0, 200);
        }
        if (typeof window.onmousemove === 'function') {
            result.winHandlerSrc = window.onmousemove.toString().substring(0, 200);
        }

        return result;
    """)
    for k, v in sorted(handler_info.items()):
        print(f"  {k}: {v}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== DONE ===")
browser.driver.quit()
