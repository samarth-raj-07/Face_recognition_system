from ultralytics import YOLO
import cv2

class FaceDetector:
    def __init__(self, conf=0.35):
        self.model = YOLO("yolov8s.pt")
        self.conf  = conf

    def detect(self, image_bgr):
        results1 = self.model(image_bgr, conf=self.conf, verbose=False, imgsz=640)[0]
        results2 = self.model(image_bgr, conf=self.conf, verbose=False, imgsz=1280)[0]

        all_boxes  = list(results1.boxes) + list(results2.boxes)
        detections = []
        seen       = []

        for box in all_boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Skip duplicate boxes (from multi-scale)
            duplicate = False
            for sx1, sy1, sx2, sy2 in seen:
                if abs(x1-sx1) < 20 and abs(y1-sy1) < 20:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen.append((x1, y1, x2, y2))

            pad = 10
            h, w = image_bgr.shape[:2]
            x1p  = max(0, x1 - pad)
            y1p  = max(0, y1 - pad)
            x2p  = min(w, x2 + pad)
            y2p  = min(h, y2 + pad)

            crop = image_bgr[y1p:y2p, x1p:x2p]
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'conf': float(box.conf[0]),
                'crop': crop
            })

        return detections