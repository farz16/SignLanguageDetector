import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset/emotions"
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 15

CLASSES = [
    "ANGRY",
    "HAPPY",
    "SAD",
    "SURPRISE",
    "NEUTRAL"
]

# ==============================
# LOAD DATASET
# ==============================

images = []
labels = []

print("Loading emotion dataset...")

for label, emotion in enumerate(CLASSES):

    folder = os.path.join(DATASET_PATH, emotion)

    if not os.path.exists(folder):
        print("Folder not found:", folder)
        continue

    count = 0

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        images.append(img)
        labels.append(label)

        count += 1

    print(emotion, ":", count, "images")


# ==============================
# PREPARE DATA
# ==============================

X = np.array(images, dtype=np.float32)
y = np.array(labels)

X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X = X / 255.0

print("\nTotal images:", len(X))

# ==============================
# TRAIN / VALIDATION SPLIT
# ==============================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training images:", len(X_train))
print("Validation images:", len(X_val))


# ==============================
# CNN MODEL
# ==============================

model = models.Sequential([

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 1)
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),

    layers.Dropout(0.5),

    layers.Dense(5, activation="softmax")
])


# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==============================
# TRAIN
# ==============================

print("\nStarting emotion model training...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==============================
# SAVE MODEL
# ==============================

model.save("emotion_model.keras")

print("\n================================")
print("Emotion model training complete!")
print("Model saved as: emotion_model.keras")
print("================================")