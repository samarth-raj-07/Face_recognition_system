import cv2
import os
from face_detector import FaceDetector

INPUT_DIR  = "data/InputData"
OUTPUT_DIR = "labeled_crops"
os.makedirs(OUTPUT_DIR, exist_ok=True)

detector     = FaceDetector()
crop_counter = {}

for img_name in sorted(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)
    image    = cv2.imread(img_path)
    detections = detector.detect(image)

    print(f"\n[{img_name}] — {len(detections)} faces detected")

    for i, det in enumerate(detections):
        crop    = det['crop']
        display = cv2.resize(crop, (224, 224))

        cv2.imshow(f"Face {i+1}/{len(detections)} — Press 1-7 to label, 0 to skip", display)
        key = cv2.waitKey(0) & 0xFF
        cv2.destroyAllWindows()

        if key in [ord(str(k)) for k in range(1, 8)]:
            person_id = chr(key)
            save_dir  = os.path.join(OUTPUT_DIR, f"person_{person_id}")
            os.makedirs(save_dir, exist_ok=True)
            crop_counter[person_id] = crop_counter.get(person_id, 0) + 1
            save_path = os.path.join(save_dir,
                        f"{person_id}_{crop_counter[person_id]:03d}.jpg")
            cv2.imwrite(save_path, crop)
            print(f"  Saved → {save_path}")
        else:
            print(f"  Skipped face {i+1}")

print("\nLabeling complete!")