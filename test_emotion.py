import cv2
import numpy as np
import tensorflow as tf

# ==============================
# LOAD EMOTION MODEL
# ==============================

model = tf.keras.models.load_model("emotion_model.keras")

emotions = [
    "ANGRY",
    "HAPPY",
    "SAD",
    "SURPRISE",
    "NEUTRAL"
]

print("Emotion model loaded successfully.")


# ==============================
# FACE DETECTOR
# ==============================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("Face detector could not be loaded.")
    exit()

print("Face detector loaded successfully.")


# ==============================
# CAMERA
# ==============================

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

print("Emotion camera started.")
print("Press Q to quit.")


# ==============================
# MAIN LOOP
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Extract face
        face = gray_frame[
            y:y+h,
            x:x+w
        ]

        if face.size == 0:
            continue

        # Resize to model input
        face = cv2.resize(
            face,
            (48, 48)
        )

        # Normalize
        face = face.astype(
            "float32"
        ) / 255.0

        # Shape: (1,48,48,1)
        face = face.reshape(
            1,
            48,
            48,
            1
        )

        # Prediction
        prediction = model.predict(
            face,
            verbose=0
        )[0]

        index = int(
            np.argmax(prediction)
        )

        emotion = emotions[index]

        confidence = (
            float(prediction[index]) * 100
        )

        # Face box
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Emotion text
        text = f"{emotion} {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # Display
    cv2.imshow(
        "Emotion Detection",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()

print("Emotion camera stopped.")