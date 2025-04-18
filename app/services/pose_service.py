import time
from app.utils.image import base64_to_pil, pil_to_cv2, cv2_to_pil, pil_to_base64
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

        keypoints_list, _ = self.detector.predict(cv2_img)

        for keypoints in keypoints_list:
            for i, (x, y, c) in enumerate(keypoints):
                if c > 0.5:
                    cv2.circle(cv2_img, (int(x), int(y)), 5, (0, 255, 0), -1)
                    cv2.putText(cv2_img, str(i), (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            connections = [[5, 6], [5, 11], [6, 12], [11, 12]]
            for c1, c2 in connections:
                if keypoints[c1][2] > 0.5 and keypoints[c2][2] > 0.5:
                    pt1 = (int(keypoints[c1][0]), int(keypoints[c1][1]))
                    pt2 = (int(keypoints[c2][0]), int(keypoints[c2][1]))
                    cv2.line(cv2_img, pt1, pt2, (0, 0, 255), 2)

        pil_result = cv2_to_pil(cv2_img)
        result_base64 = pil_to_base64(pil_result)

        return {
            "id": data.id,
            "image": result_base64
        }
