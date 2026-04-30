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
IMG_SIZE = (128, 128)

os.makedirs(FRAME_DIR, exist_ok=True)

# ===== STATE =====
key_state = {}
mouse_state = {
    "x": 0,
    "y": 0,
    "left": 0,
    "right": 0
}

# ===== KEY MAPPING =====
TRACKED_KEYS = {
    "w": "up",
    "a": "left",
    "s": "down",
    "d": "right",
    "space": "jump",
    "shift": "run"
}

# ===== KEYBOARD LISTENERS =====
def on_press(key):
    global running, recording
    try:
        k = key.char.lower()
    except:
        k = str(key).replace("Key.", "")

    # Quit
    if k == 'q':
        print("Stopping...")
        running = False
        return False

    # Toggle recording
    if k == 'r':
        recording = not recording
        print("Recording:", recording)
        if recording:
            beep(1200, 200)  # start sound
        else:
            beep(600, 200)   # stop sound

    if k in TRACKED_KEYS:
        key_state[TRACKED_KEYS[k]] = 1

def on_release(key):
    try:
        k = key.char.lower()
    except:
        k = str(key).replace("Key.", "")

    if k in TRACKED_KEYS:
        key_state[TRACKED_KEYS[k]] = 0

# ===== MOUSE LISTENERS =====
def on_move(x, y):
    mouse_state["x"] = x
    mouse_state["y"] = y

def on_click(x, y, button, pressed):
    if button == mouse.Button.left:
        mouse_state["left"] = int(pressed)
    elif button == mouse.Button.right:
        mouse_state["right"] = int(pressed)

def beep(frequency=1000, duration=150):
    winsound.Beep(frequency, duration)

# ===== START LISTENERS =====
keyboard.Listener(on_press=on_press, on_release=on_release).start()
mouse.Listener(on_move=on_move, on_click=on_click).start()

# ===== SCREEN CAPTURE =====
sct = mss()

# Capture full screen (you can crop later if needed)
monitor = sct.monitors[1]

# ===== MAIN LOOP =====
data_file = open(os.path.join(SAVE_DIR, "data.jsonl"), "w")

frame_id = 0
frame_time = 1.0 / FPS

print("Recording... Press CTRL+C to stop.")

try:
    while True:
        start = time.time()

        # Capture screen
        screenshot = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

        # Resize for ML
        frame_resized = cv2.resize(frame, IMG_SIZE)

        # Save image
        frame_name = f"frame_{frame_id:06d}.jpg"
        frame_path = os.path.join(FRAME_DIR, frame_name)
        cv2.imwrite(frame_path, frame_resized)

        # Timestamp
        ts = time.time()

        # Save metadata
        record = {
            "frame": frame_name,
            "timestamp": ts,
            "keys": key_state.copy(),
            "mouse": mouse_state.copy()
        }

        data_file.write(json.dumps(record) + "\n")

        frame_id += 1

        # Maintain FPS
        elapsed = time.time() - start
        time.sleep(max(0, frame_time - elapsed))

except KeyboardInterrupt:
    print("\nStopped.")

finally:
    data_file.close()