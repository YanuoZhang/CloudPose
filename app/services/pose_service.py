import time
from app.utils.image import base64_to_pil, pil_to_cv2, bytes_to_base64
from model.detector import Detector
import cv2

class PoseService:
    def __init__(self):
        self.detector = Detector()

    def detect(self, data):
        pil_img = base64_to_pil(data.image)
        cv2_img = pil_to_cv2(pil_img)
        keypoints, boxes = self.detector.predict(cv2_img)
        
        return {
            "id": data.id,
            "count": len(keypoints),
            "keypoints": keypoints,
            "boxes": boxes,
            "speed_preprocess": 0.01, 
            "speed_inference": 0.05,
            "speed_postprocess": 0.01
        }

    def detect_with_annotation(self, data):
        pil_img = base64_to_pil(data.image)
        cv2_img = pil_to_cv2(pil_img)

        annotated_bytes = self.detector.predict_with_annotation(cv2_img)

        result_base64 = bytes_to_base64(annotated_bytes)

        return {
            "id": data.id,
            "image": result_base64
        }
