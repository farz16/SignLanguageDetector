import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def finger_up(tip, pip, lm):
    return lm[tip].y < lm[pip].y


while True:
    success, img = cap.read()

    if not success:
        break

    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    gesture = "No Hand"

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            lm = hand.landmark

            index = finger_up(8, 6, lm)
            middle = finger_up(12, 10, lm)
            ring = finger_up(16, 14, lm)
            pinky = finger_up(20, 18, lm)

            if index and middle and ring and pinky:
                gesture = "OPEN PALM"

            elif not index and not middle and not ring and not pinky:
                gesture = "FIST"

            else:
                gesture = "UNKNOWN"


    cv2.putText(
        img,
        gesture,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0,255,0),
        3
    )

    cv2.imshow("Sign Language Detector", img)


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()