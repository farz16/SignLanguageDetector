import cv2
import mediapipe as mp
import os
import time

# =========================================================
# 1. WAHI 17 SIGNS KI LIST (Jo pehle tay kiye the)
# =========================================================
NEW_SIGNS = [
    ("BAD", "Thumbs Down (Bura/Dislike)"),
    ("PEACE", "V-Shape Index + Middle"),
    ("OKAY", "Index + Thumb Circle (Circle Shape)"),
    ("I LOVE YOU", "Thumb + Index + Pinky Up"),
    ("CALL ME", "Thumb + Pinky Out (Phone Sign)"),
    ("ROCK ON", "Index + Pinky Up (Horns Shape)"),
    ("ONE", "Only Index Finger Up (Number 1)"),
    ("THREE", "Index + Middle + Ring Up (Number 3)"),
    ("FOUR", "4 Fingers Up, Thumb Folded"),
    ("LUCK", "Index + Middle Crossed (Fingers Crossed)"),
    ("PROMISE", "Only Pinky Finger Up (Chhoti Ungli)"),
    ("LITTLE", "Pinch Gesture (Aadha inch gap)"),
    ("HEART", "Finger Heart (Thumb + Index Cross)"),
    ("WATER", "W-Shape (3 Fingers Spread)"),
    ("NO", "Thumb + Index L-Shape"),
    ("COME HERE", "Index Finger Curved Forward"),
    ("TIME", "Flat Hand Wrist Tap")
]

SAMPLES_PER_SIGN = 100

# Camo Studio Camera (Index 1)
cap = cv2.VideoCapture(1)

# Camo warm-up
for _ in range(10):
    cap.read()

# MediaPipe Setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

print("\n" + "="*60)
print("🚀 CAMO DATASET COLLECTOR: 17 SIGNS")
print("="*60 + "\n")

for sign_name, hint in NEW_SIGNS:
    folder = os.path.join("dataset", sign_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # --- Phase 1: 3-Second Countdown (Ready hone ke liye) ---
    start_countdown = time.time()
    while True:
        elapsed = time.time() - start_countdown
        remaining = 3 - int(elapsed)

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

        frame = cv2.flip(frame, 1)

        # Skeleton Preview
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # Info Box
        cv2.rectangle(frame, (10, 10), (630, 100), (20, 24, 35), -1)
        cv2.rectangle(frame, (10, 10), (630, 100), (0, 230, 118), 2)
        cv2.putText(frame, f"TARGET: [{sign_name}]", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {hint}", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Countdown
        if remaining > 0:
            cv2.putText(frame, f"GET READY IN: {remaining}s...", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "RECORDING STARTING...", (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Sign Dataset Collector", frame)
        cv2.waitKey(20)

        if elapsed >= 3.0:
            break

    # --- Phase 2: Auto-Capture 100 Images ---
    count = 0
    while count < SAMPLES_PER_SIGN:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            img_path = os.path.join(folder, f"{count}.jpg")
            cv2.imwrite(img_path, frame)
            count += 1

        # Red Recording Box
        cv2.rectangle(frame, (10, 10), (630, 80), (0, 0, 180), -1)
        cv2.putText(frame, f"RECORDING [{sign_name}]: {count}/{SAMPLES_PER_SIGN}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        bar_w = int((count / SAMPLES_PER_SIGN) * 600)
        cv2.rectangle(frame, (20, 58), (20 + bar_w, 68), (0, 255, 0), -1)

        cv2.imshow("Sign Dataset Collector", frame)
        cv2.waitKey(15)

    print(f"✓ '{sign_name}' (100 images) saved successfully!")
    time.sleep(0.4)

cap.release()
hands.close()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("🎉 ALL 17 SIGNS COLLECTED SUCCESSFULLY!")
print("👉 Ab terminal me run karein: py train.py")
print("="*60 + "\n")