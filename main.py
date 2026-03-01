import cv2
import numpy as np
import time

# ===============================
# DETECTION SETTINGS
# ===============================
STD_MULTIPLIER = 2.0
MIN_AREA = 300
STATE_STABLE_TIME = 0.15

# ===============================
# MORSE SETTINGS (RATIO BASED)
# ===============================
IDLE_TIMEOUT_MULT = 7

MORSE_DECODE = {
    '.-':'A','-...':'B','-.-.':'C','-..':'D',
    '.':'E','..-.':'F','--.':'G','....':'H',
    '..':'I','.---':'J','-.-':'K','.-..':'L',
    '--':'M','-.':'N','---':'O','.--.':'P',
    '--.-':'Q','.-.':'R','...':'S','-':'T',
    '..-':'U','...-':'V','.--':'W','-..-':'X',
    '-.--':'Y','--..':'Z'
}

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed")
    exit()

print("Receiver running (energy-weighted tracking)")
print("Press ESC to quit")

light_on = False
last_change_time = time.time()
state_candidate_time = time.time()

dot_time = None
current_symbol = ""
current_word = ""

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    now = time.time()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    mean_val = np.mean(gray)
    std_val = np.std(gray)
    dynamic_thresh = mean_val + STD_MULTIPLIER * std_val

    _, thresh = cv2.threshold(gray, dynamic_thresh, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # ===============================
    # ENERGY-WEIGHTED BLOB SELECTION
    # ===============================
    best_blob = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        mean_intensity = cv2.mean(gray, mask=mask)[0]
        score = area * mean_intensity

        if score > best_score:
            best_score = score
            best_blob = cnt

    detected = False

    if best_blob is not None:
        x,y,w,h = cv2.boundingRect(best_blob)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        detected = True

    # ===============================
    # DEBOUNCE FILTER
    # ===============================
    if detected != light_on:
        if now - state_candidate_time > STATE_STABLE_TIME:
            prev_light = light_on
            light_on = detected
            duration = now - last_change_time
            last_change_time = now

            # ===============================
            # MORSE TIMING
            # ===============================
            if not light_on:

                if dot_time is None:
                    if 0.5 < duration < 1.5:
                        dot_time = duration
                        print(f"[LOCKED] dot_time ≈ {dot_time:.2f}s")
                    else:
                        print("Rejected first pulse:", round(duration,2))
                    continue

                ratio = duration / dot_time

                if 0.6 <= ratio <= 1.6:
                    current_symbol += '.'
                    print("DOT")
                elif 2.0 <= ratio <= 4.0:
                    current_symbol += '-'
                    print("DASH")
                else:
                    print("Ignored pulse:", round(duration,2))

            else:
                if dot_time is not None:
                    if duration > dot_time * 3:
                        if current_symbol:
                            letter = MORSE_DECODE.get(current_symbol, '?')
                            current_word += letter
                            print("Decoded:", current_word)
                            current_symbol = ""

    else:
        state_candidate_time = now

    # ===============================
    # IDLE FLUSH
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

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()