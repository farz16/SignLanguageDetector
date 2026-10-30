import cv2
import mediapipe as mp
import os
import numpy as np


word = input("Enter word name: ")

folder = "dataset/" + word

os.makedirs(folder, exist_ok=True)


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)

count = 0

print("Press SPACE to collect data")
print("Press ESC to exit")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)


    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )


            data = []

            for point in hand.landmark:
                data.append(point.x)
                data.append(point.y)


            key = cv2.waitKey(1) & 0xFF


            if key == 32:   # SPACE

                np.save(
                    f"{folder}/{count}.npy",
                    np.array(data)
                )

                count += 1
                print("Saved:", count)


    cv2.putText(
        frame,
        f"{word}: {count}/200",
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )


    cv2.imshow(
        "Collect Word Data",
        frame
    )


    if count >= 200:
        break


    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
