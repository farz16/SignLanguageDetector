from datasets import load_dataset
import os
from PIL import Image

# Dataset download
print("Downloading FER2013 dataset...")

dataset = load_dataset("AutumnQiu/fer2013")

print("Dataset loaded successfully!")

# Our required emotions
emotion_names = {
    0: "ANGRY",
    3: "HAPPY",
    4: "SAD",
    5: "SURPRISE",
    6: "NEUTRAL"
}

# Create folders
base_path = "dataset/emotions"

for emotion in emotion_names.values():
    os.makedirs(os.path.join(base_path, emotion), exist_ok=True)

count = {
    "ANGRY": 0,
    "HAPPY": 0,
    "SAD": 0,
    "SURPRISE": 0,
    "NEUTRAL": 0
}

# Process dataset
for split in dataset:

    print("Processing:", split)

    for item in dataset[split]:

        label = item["label"]

        if label not in emotion_names:
            continue

        emotion = emotion_names[label]

        image = item["image"]

        filename = f"{emotion}_{count[emotion]}.jpg"

        save_path = os.path.join(
            base_path,
            emotion,
            filename
        )

        image.convert("RGB").save(save_path)

        count[emotion] += 1

print("\nDataset preparation completed!")

for emotion in count:
    print(emotion, ":", count[emotion], "images")