import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# Load Age Model
# ==============================
model = load_model("age_model.keras")

AGE_CLASSES = [
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55+"
]

# ==============================
# Face Detector
# ==============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==============================
# Camera
# ==============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Age Detection Started...")
print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        # Crop face
        face = gray[y:y+h, x:x+w]

        # Resize to model input
        face = cv2.resize(face, (48, 48))

        # Normalize
        face = face.astype("float32") / 255.0

        # Add dimensions
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        # Predict
        prediction = model.predict(face, verbose=0)

        index = np.argmax(prediction)
        age = AGE_CLASSES[index]
        confidence = prediction[0][index] * 100

        # Face rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Age text
        text = f"Age: {age} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    cv2.imshow("Age Detection Test", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
