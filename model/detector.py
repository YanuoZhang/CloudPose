from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
from model.pose_detection import predict as pose_predict 


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
                xy = keypoints.xy[0].cpu().numpy()
                conf = keypoints.conf[0].cpu().numpy()
                all_keypoints.append([[float(x), float(y), float(c)] for (x, y), c in zip(xy, conf)])

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    prob = float(box.conf[0].cpu().numpy())
                    all_boxes.append({
                        "x": float(x1), "y": float(y1),
                        "width": float(w), "height": float(h),
                        "probability": prob
                    })

        return all_keypoints, all_boxes

    def predict_with_annotation(self, image: np.ndarray) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_input:
            input_path = tmp_input.name
            cv2.imwrite(input_path, image)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_output:
            output_path = tmp_output.name

        pose_predict(self.model, input_path, output_path)

        with open(output_path, "rb") as f:
            annotated_bytes = f.read()

        return annotated_bytes
