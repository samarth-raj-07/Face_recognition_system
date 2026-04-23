import os
import pickle
import numpy as np
import cv2
from face_embedder import FaceEmbedder

LABELED_DIR  = "labeled_crops"
GALLERY_PATH = "gallery/gallery.pkl"
os.makedirs("gallery", exist_ok=True)

embedder = FaceEmbedder()
gallery  = {}

print("Processing labeled crops...")
for person_folder in sorted(os.listdir(LABELED_DIR)):
    person_path = os.path.join(LABELED_DIR, person_folder)
    if not os.path.isdir(person_path):
        continue

    embeddings = []
    for img_file in os.listdir(person_path):
        img_path = os.path.join(person_path, img_file)
        img      = cv2.imread(img_path)
        if img is None:
            continue
        emb = embedder.get_embedding_from_crop(img)
        if emb is not None:
            embeddings.append(emb)

    if embeddings:
        gallery[person_folder] = embeddings
        print(f"  {person_folder}: {len(embeddings)} embeddings")

with open(GALLERY_PATH, "wb") as f:
    pickle.dump(gallery, f)

print(f"\nGallery saved → {GALLERY_PATH}")
print(f"Total identities: {len(gallery)}")