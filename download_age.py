import os
import urllib.request
import zipfile

# Dataset URL
url = "https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new"

zip_path = "utkface.zip"
extract_path = "utkface_data"

print("Downloading age dataset...")
urllib.request.urlretrieve(url, zip_path)

print("Download complete!")

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete!")

# Remove ZIP file
os.remove(zip_path)

print("Age dataset is ready.")