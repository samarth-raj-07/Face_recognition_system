import cv2
import numpy as np
import pickle
import os
import time
from face_detector import FaceDetector
from face_embedder import FaceEmbedder

GALLERY_PATH = "gallery/gallery.pkl"
TEST_DIR     = "data/Test_512_512"
OUTPUT_DIR   = "output"
THRESHOLD    = 0.35
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
detector = FaceDetector()
embedder = FaceEmbedder()

with open(GALLERY_PATH, "rb") as f:
    gallery = pickle.load(f)

gallery_centroids = {
    person: np.mean(embs, axis=0)
    for person, embs in gallery.items()
}

# ── Matching ──────────────────────────────────────────────────────────────────
def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def match_identity(embedding):
    best_name  = "Unknown"
    best_score = -1.0
    for person, centroid in gallery_centroids.items():
        score = cosine_similarity(embedding, centroid)
        if score > best_score:
            best_score = score
            best_name  = person
    if best_score < THRESHOLD:
        return "Unknown", best_score
    return best_name, best_score

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,255),
    (0,255,255),(255,165,0),(128,0,128),(255,50,50)
]
person_colors = {}

def get_color(name):
    if name not in person_colors:
        person_colors[name] = COLORS[len(person_colors) % len(COLORS)]
    return person_colors[name]

# ── Process image ─────────────────────────────────────────────────────────────
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot read: {image_path}")
        return

    start      = time.time()
    detections = detector.detect(image)

    results = []
    for det in detections:
        emb = embedder.get_embedding_from_crop(det['crop'])
        if emb is None:
            name, score = "Unknown", 0.0
        else:
            name, score = match_identity(emb)
        results.append({'name': name, 'score': score, 'bbox': det['bbox']})

    elapsed   = time.time() - start
    annotated = image.copy()

    for r in results:
        x1, y1, x2, y2 = r['bbox']
        name  = r['name']
        score = r['score']
        color = get_color(name)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label       = f"{name} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1-th-10), (x1+tw+4, y1), color, -1)
        cv2.putText(annotated, label, (x1+2, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(annotated,
                f"Faces: {len(results)}  |  Time: {elapsed:.3f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)

    fname    = os.path.basename(image_path)
    out_path = os.path.join(OUTPUT_DIR, f"result_{fname}")
    cv2.imwrite(out_path, annotated)

    print(f"\n[{fname}]  Detected: {len(results)} faces  |  Time: {elapsed:.3f}s")
    for r in results:
        print(f"   {r['name']:15s}  score={r['score']:.3f}  bbox={r['bbox']}")
    print(f"   Saved → {out_path}")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for fname in sorted(os.listdir(TEST_DIR)):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            process_image(os.path.join(TEST_DIR, fname))