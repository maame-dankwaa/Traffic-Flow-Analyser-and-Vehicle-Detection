import cv2
import os
from zipfile import ZipFile
from pathlib import Path

#os.mkdir("data", exist_ok=True)

#Get the training images folder
#with ZipFile("VehiclesDetectionDataset.zip","r") as zip_ref:
    #zip_ref.extractall()

main_dir = Path("data/train/images")
target_dir = Path("data/processed")
target_dir.mkdir(parents=True, exist_ok=True)

#Copy each image file into the processed folder
for img_file in main_dir.glob("*.jpg"):
    print(f"Processing image: {img_file.name}")
    img = cv2.imread(str(img_file))

    if img is not None:
        resized_img = cv2.resize(img, (640, 640))
        save_path = target_dir / img_file.name
        cv2.imwrite(str(save_path), resized_img)
        print(f"Saved to {save_path}")