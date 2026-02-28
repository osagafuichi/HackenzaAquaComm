import cv2
import numpy as np
import time

# Morse map
MORSE_DECODE = {
    '.-':'A','-...':'B','-.-.':'C','-..':'D',
    '.':'E','..-.':'F','--.':'G','....':'H',
    '..':'I','.---':'J','-.-':'K','.-..':'L',
    '--':'M','-.':'N','---':'O','.--.':'P',
    '--.-':'Q','.-.':'R','...':'S','-':'T',
    '..-':'U','...-':'V','.--':'W','-..-':'X',
    '-.--':'Y','--..':'Z'
}

# ===============================
# PARAMETERS (TUNED FOR 1s DOT)
# ===============================
ROI_SIZE = 30
IDLE_TIMEOUT_MULT = 7
TRACK_SMOOTH = 0.7

MIN_DOT_TIME = 0.8   # 🔥 prevents early false lock
MAX_DOT_TIME = 1.2   # 🔥 keeps timing sane

# ===============================
# FIND BRIGHTEST GRID CELL
# ===============================
def find_brightest_cell(gray, cell_size):
    h, w = gray.shape
    best_val = -1
    best_coords = (0, 0)

    for y in range(0, h - cell_size, cell_size):
        for x in range(0, w - cell_size, cell_size):
            cell = gray[y:y+cell_size, x:x+cell_size]
            mean_val = np.mean(cell)

            if mean_val > best_val:
                best_val = mean_val
                best_coords = (x, y)

    return best_coords, best_val


cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("❌ Camera failed to open")
    exit()

print("Receiver running... (auto-track mode)")
print("⚙️  Dot ≈ 1 sec, Dash ≈ 3 sec")

light_on = False
last_change_time = time.time()
light_on_start = None

current_symbol = ""
current_word = ""

dot_time = None
brightness_avg = None
tracked_pos = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    # ===============================
    # AUTO LIGHT TRACKING
    # ===============================
    (best_x, best_y), _ = find_brightest_cell(gray, ROI_SIZE)

    if tracked_pos is None:
        tracked_pos = (best_x, best_y)
    else:
        tx, ty = tracked_pos
        tx = int(TRACK_SMOOTH * tx + (1 - TRACK_SMOOTH) * best_x)
        ty = int(TRACK_SMOOTH * ty + (1 - TRACK_SMOOTH) * best_y)
        tracked_pos = (tx, ty)

    x1, y1 = tracked_pos
    x2 = x1 + ROI_SIZE
    y2 = y1 + ROI_SIZE

    roi = gray[y1:y2, x1:x2]
    brightness = float(np.mean(roi))

    # ===============================
    # SMOOTH BRIGHTNESS
    # ===============================
    if brightness_avg is None:
        brightness_avg = brightness
    else:
        brightness_avg = 0.92 * brightness_avg + 0.08 * brightness

    thresh_on = brightness_avg + 30
    thresh_off = brightness_avg + 8

    prev_light = light_on

    # ===============================
    # HYSTERESIS LIGHT DETECTION
    # ===============================
    if not light_on and brightness > thresh_on:
        light_on = True
        light_on_start = now

    elif light_on:
        min_on_time = (dot_time * 0.6) if dot_time else 0.4

        if brightness < thresh_off and (now - light_on_start) > min_on_time:
            light_on = False

    # ===============================
    # EDGE TIMING
    # ===============================
    if light_on != prev_light:
        duration = now - last_change_time
        last_change_time = now

        # ignore flicker
        if dot_time is not None and duration < dot_time * 0.45:
            continue

        # ----- pulse ended -----
        if not light_on:

            if dot_time is None:
                # 🔥 prevent early bad lock
                if duration < MIN_DOT_TIME:
                    continue

                dot_time = duration
                print(f"[LOCKED] dot_time ≈ {dot_time:.2f}s")

            else:
                if duration < dot_time * 1.35:
                    dot_time = 0.97 * dot_time + 0.03 * duration
                    dot_time = max(MIN_DOT_TIME, min(dot_time, MAX_DOT_TIME))

            ratio = duration / dot_time

            if ratio < 1.9:
                current_symbol += '.'
            else:
                current_symbol += '-'

        # ----- gap ended -----
        else:
            if dot_time is not None:
                letter_gap = dot_time * 3
                word_gap = dot_time * 7

                if duration > letter_gap:
                    if current_symbol:
                        letter = MORSE_DECODE.get(current_symbol, '?')
                        current_word += letter
                        print("Decoded:", current_word)
                        current_symbol = ""

                if duration > word_gap:
                    if current_word:
                        print("Word complete:", current_word)
                        current_word = ""

    # ===============================
    # IDLE WORD FLUSH
    # ===============================
    if not light_on and dot_time is not None:
        idle_time = now - last_change_time

        if idle_time > dot_time * IDLE_TIMEOUT_MULT:
            if current_symbol:
                letter = MORSE_DECODE.get(current_symbol, '?')
                current_word += letter
                current_symbol = ""

            if current_word:
                print("Word complete:", current_word)
                current_word = ""

            last_change_time = now

    # draw tracking box
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imshow("Receiver", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
