import os
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

print("Loading age dataset...")

DATASET_PATH = "dataset/age"

IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 20

if not os.path.exists(DATASET_PATH):
    print("ERROR: Age dataset folder not found!")
    print("Expected:", DATASET_PATH)
    exit()

# -------------------------------------------------
# DATA AUGMENTATION
# -------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("\nAge classes:")
print(train_data.class_indices)

print("\nTraining images:", train_data.samples)
print("Validation images:", val_data.samples)

# -------------------------------------------------
# CLASS WEIGHTS
# -------------------------------------------------

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(
    enumerate(class_weights_array)
)

print("\nClass weights:")
print(class_weights)

# -------------------------------------------------
# CNN MODEL
# -------------------------------------------------

model = models.Sequential([
    layers.Input(shape=(48, 48, 1)),

    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------------------------
# CALLBACKS
# -------------------------------------------------

early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    mode="max",
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

checkpoint = callbacks.ModelCheckpoint(
    "age_model.keras",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

# -------------------------------------------------
# TRAINING
# -------------------------------------------------

print("\n================================")
print("Starting improved age model training...")
print("================================\n")

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)

print("\n================================")
print("Age model training complete!")
print("Best model saved as: age_model.keras")
print("================================")