import os
import shutil

# Downloaded UTKFace dataset
source_root = "utkface_data"

# Our age folders
output_root = os.path.join("dataset", "age")

age_groups = {
    "18-24": (18, 24),
    "25-34": (25, 34),
    "35-44": (35, 44),
    "45-54": (45, 54),
    "55-plus": (55, 120)
}

# Make sure folders exist
for folder in age_groups:
    os.makedirs(os.path.join(output_root, folder), exist_ok=True)

count = {folder: 0 for folder in age_groups}

print("Organizing age images...")

# Search all files inside extracted dataset
for root, dirs, files in os.walk(source_root):

    for filename in files:

        # UTKFace files look like:
        # 25_0_0_201701171...
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        try:
            age = int(filename.split("_")[0])
        except:
            continue

        # Find correct age group
        selected_group = None

        for group, (minimum, maximum) in age_groups.items():
            if minimum <= age <= maximum:
                selected_group = group
                break

        if selected_group is None:
            continue

        source_file = os.path.join(root, filename)
        destination_file = os.path.join(
            output_root,
            selected_group,
            filename
        )

        # Copy image
        if not os.path.exists(destination_file):
            shutil.copy2(source_file, destination_file)
            count[selected_group] += 1

print("\n================================")
print("Age dataset organization complete!")
print("================================")

for group in age_groups:
    print(group, ":", count[group], "images")