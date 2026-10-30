import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier


# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7
)


# =========================
# DATA
# =========================

data = []
labels = []

dataset_path = "dataset"


# =========================
# READ DATASET
# =========================

for label in os.listdir(dataset_path):

    folder = os.path.join(dataset_path, label)

    if not os.path.isdir(folder):
        continue

    print("Processing:", label)

    for img_name in os.listdir(folder):

        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:

            landmarks = []

            for point in result.multi_hand_landmarks[0].landmark:

                landmarks.append(point.x)
                landmarks.append(point.y)

            data.append(landmarks)
            labels.append(label)


# =========================
# CHECK DATA
# =========================

print("Total samples:", len(data))
print("Classes:", sorted(set(labels)))


# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(data, labels)


# =========================
# SAVE MODEL
# =========================

with open("model.pkl", "wb") as f:

    pickle.dump(model, f)


print("Model saved successfully!")