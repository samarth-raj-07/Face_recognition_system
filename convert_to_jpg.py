# convert_to_jpg.py
import cv2
import os

ATT_DIR = r"K:\archive"

for folder in sorted(os.listdir(ATT_DIR)):
    folder_path = os.path.join(ATT_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    for img_file in os.listdir(folder_path):
        if not img_file.endswith(".pgm"):
            continue

        pgm_path = os.path.join(folder_path, img_file)
        jpg_path = pgm_path.replace(".pgm", ".jpg")

        img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(jpg_path, img)
        os.remove(pgm_path)   # delete original pgm

        print(f"Converted: {jpg_path}")

print("\nAll done — all PGM converted to JPG")