# dataset_preparation.py
import cv2
import numpy as np
import os
import random

DATA_DIR    = "data"               # S1 to S7 are directly inside here
INPUT_DIR   = "data/InputData"
TEST_DIR    = "data/Test_512_512"
CANVAS_SIZE = 512
FACE_SIZE   = 112

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TEST_DIR,  exist_ok=True)

random.seed(42)

# ── Convert PGM to JPG if not already done ───────────────────────────────
for folder in ["S1","S2","S3","S4","S5","S6","S7"]:
    fpath = os.path.join(DATA_DIR, folder)
    for f in os.listdir(fpath):
        if f.endswith(".pgm"):
            pgm = os.path.join(fpath, f)
            jpg = pgm.replace(".pgm", ".jpg")
            img = cv2.imread(pgm, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(jpg, img)
            os.remove(pgm)
print("PGM → JPG done (skipped if already JPG)")

# ── Load images per subject ───────────────────────────────────────────────
subjects = ["S1","S2","S3","S4","S5","S6","S7"]

subject_images = {}
for s in subjects:
    folder = os.path.join(DATA_DIR, s)
    files  = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
    subject_images[s] = [os.path.join(folder, f) for f in files]
    print(f"  {s}: {len(files)} images found")

# ── Split images per subject ──────────────────────────────────────────────
# First 7 images → used for InputData (labeling + gallery)
# Last 3 images  → used for Test set
gallery_imgs = {s: subject_images[s][:7] for s in subjects}
test_imgs    = {s: subject_images[s][7:] for s in subjects[:5]}  # S1-S5 in test

# ── Helper: create composite image ───────────────────────────────────────
def make_composite(chosen, img_pool):
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

    positions = [
        (30,  30),
        (200, 30),
        (370, 30),
        (110, 260),
        (300, 260),
    ][:len(chosen)]

    for subject, (x, y) in zip(chosen, positions):
        img_path = random.choice(img_pool[subject])
        face     = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        face     = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
        face_bgr = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        canvas[y:y+FACE_SIZE, x:x+FACE_SIZE] = face_bgr

    return canvas

# ── Create InputData (Val_01 to Val_08) ──────────────────────────────────
print("\nCreating InputData...")
for i in range(1, 9):
    num_faces = random.choice([4, 5])
    chosen    = random.sample(subjects, num_faces)
    canvas    = make_composite(chosen, gallery_imgs)
    out       = os.path.join(INPUT_DIR, f"Val_{i:02d}.jpg")
    cv2.imwrite(out, canvas)
    print(f"  Val_{i:02d}.jpg — {num_faces} faces — {chosen}")

# ── Create Test set (Test_01 to Test_10) ─────────────────────────────────
print("\nCreating Test_512_512...")
test_subjects = subjects[:5]   # S1 to S5 only
for i in range(1, 11):
    num_faces = random.choice([4, 5])
    chosen    = random.sample(test_subjects, min(num_faces, 5))
    canvas    = make_composite(chosen, test_imgs)
    out       = os.path.join(TEST_DIR, f"Test_{i:02d}.jpg")
    cv2.imwrite(out, canvas)
    print(f"  Test_{i:02d}.jpg — {len(chosen)} faces — {chosen}")

# ── Summary ───────────────────────────────────────────────────────────────
print("\n✅ Done!")
print(f"  InputData    → {INPUT_DIR}  (8 images, 7 persons S1–S7)")
print(f"  Test_512_512 → {TEST_DIR}  (10 images, 5 persons S1–S5)")
print("\nNext steps:")
print("  python label_tool.py")
print("  python gallery_builder.py")
print("  python inference.py")