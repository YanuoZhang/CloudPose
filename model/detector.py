from ultralytics import YOLO
import numpy as np
import cv2

class Detector:
    def __init__(self, model_path: str = "model3-yolol/yolo11l-pose.pt"):
        self.model = YOLO(model_path)

    def predict(self, image: np.ndarray):
        results = self.model.predict(source=image, save=False)  

        all_keypoints = []
        all_boxes = []

        for result in results:
            keypoints = result.keypoints
            boxes = result.boxes

            if keypoints is not None and len(keypoints.xy) > 0:
                xy = keypoints.xy[0].cpu().numpy()          # (num_keypoints, 2)
                conf = keypoints.conf[0].cpu().numpy()      # (num_keypoints,)
                all_keypoints.append([[float(x), float(y), float(c)] for (x, y), c in zip(xy, conf)])

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    prob = float(box.conf[0].cpu().numpy())
                    all_boxes.append({"x": float(x1), "y": float(y1), "width": float(w), "height": float(h), "probability": prob})

        return all_keypoints, all_boxes
