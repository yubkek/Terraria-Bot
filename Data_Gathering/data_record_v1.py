import os
import time
import json
import cv2
import numpy as np
from mss import mss
from pynput import keyboard, mouse
import winsound

running = True
recording = False

# ===== CONFIG =====
SAVE_DIR = "dataset_v1"
FRAME_DIR = os.path.join(SAVE_DIR, "frames")
FPS = 12

# KEEP 16:9 (IMPORTANT FOR ML)
IMG_SIZE = (160, 90)

os.makedirs(FRAME_DIR, exist_ok=True)

# ===== STATE =====
TRACKED_KEYS = {
    "a": "move_left",
    "d": "move_right",
    "w": "move_up",
    "s": "move_down",
    "space": "jump",
    "shift": "run",

    "1": "slot_1",
    "2": "slot_2",
    "3": "slot_3",
    "4": "slot_4",
    "5": "slot_5",
    "6": "slot_6",
    "7": "slot_7",
    "8": "slot_8",
    "9": "slot_9",

    "esc": "esc",

    "h": "heal",
    "e": "grapple"
}

key_state = {v: 0 for v in TRACKED_KEYS.values()}

mouse_state = {
    "x": 0,
    "y": 0,
    "left": 0,
    "right": 0
}

# ===== SOUND =====
def beep(frequency=1000, duration=150):
    winsound.Beep(frequency, duration)

# ===== KEYBOARD =====
def on_press(key):
    global running, recording

    try:
        if hasattr(key, "char") and key.char is not None:
            k = key.char.lower()
        else:
            k = str(key).replace("Key.", "").lower()
    except:
        return

    # QUIT
    if k == "q":
        print("Stopping...")
        running = False
        return False

    # TOGGLE RECORDING
    if k == "r":
        recording = not recording
        print("Recording:", recording)
        beep(1200 if recording else 600, 200)

    # KEY DOWN
    if k in TRACKED_KEYS:
        key_state[TRACKED_KEYS[k]] = 1

def on_release(key):
    try:
        if hasattr(key, "char") and key.char is not None:
            k = key.char.lower()
        else:
            k = str(key).replace("Key.", "").lower()
    except:
        return

    if k in TRACKED_KEYS:
        key_state[TRACKED_KEYS[k]] = 0

# ===== MOUSE =====
def on_move(x, y):
    mouse_state["x"] = x
    mouse_state["y"] = y

def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        mouse_state["left"] = int(pressed)
    elif button == mouse.Button.right:
        mouse_state["right"] = int(pressed)

# ===== START LISTENERS =====
keyboard.Listener(on_press=on_press, on_release=on_release).start()
mouse.Listener(on_move=on_move, on_click=on_click).start()

# ===== SCREEN =====
sct = mss()
monitor = sct.monitors[1]

# ===== DATA FILE =====
data_file = open(os.path.join(SAVE_DIR, "data.jsonl"), "w")

frame_id = 0
frame_time = 1.0 / FPS

print("Ready. Press R to start recording.")

# ===== MAIN LOOP =====
try:
    while running:
        start = time.time()

        if recording:
            screenshot = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

            # KEEP ASPECT RATIO SAFE (16:9)
            frame_resized = cv2.resize(frame, IMG_SIZE)

            frame_name = f"frame_{frame_id:06d}.jpg"
            frame_path = os.path.join(FRAME_DIR, frame_name)
            cv2.imwrite(frame_path, frame_resized)

            record = {
                "frame": frame_name,
                "timestamp": time.time(),
                "keys": key_state.copy(),
                "mouse": mouse_state.copy()
            }

            data_file.write(json.dumps(record) + "\n")
            data_file.flush()

            frame_id += 1

        # FPS control
        elapsed = time.time() - start
        time.sleep(max(0, frame_time - elapsed))

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    data_file.close()