import cv2
import mediapipe as mp

# MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detection = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

# Camera 1
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ERROR: Camera could not open")
    exit()

print("Camera started successfully")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Frame not received")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hand detection
    hand_result = hands.process(rgb)

    # Face detection
    face_result = face_detection.process(rgb)

    # Draw hands
    if hand_result.multi_hand_landmarks:
        for hand in hand_result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    # Draw faces
    if face_result.detections:
        h, w, _ = frame.shape

        for detection in face_result.detections:
            box = detection.location_data.relative_bounding_box

            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            x = max(0, x)
            y = max(0, y)

            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        "AI CAMERA TEST",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("AI Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
face_detection.close()