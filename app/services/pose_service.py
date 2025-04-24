import time
from app.utils.image import base64_to_pil, pil_to_cv2, bytes_to_base64
from models.detector import Detector
import logging
logger = logging.getLogger(__name__)

class PoseService:
    def __init__(self):
        self.detector = Detector()

    def detect(self, data):
        pil_img = base64_to_pil(data.image)
        cv2_img = pil_to_cv2(pil_img)
        keypoints, boxes = self.detector.predict(cv2_img)
        
        start = time.time()
        # preprocess
        pil_img = base64_to_pil(data.image)
        cv2_img = pil_to_cv2(pil_img)
        t1 = time.time()
        # inference
        keypoints, boxes = self.detector.predict(cv2_img)
        t2 = time.time()
        # postprocess
        time.sleep(0.01)
        t3 = time.time()
        logger.info(
            f"[Pose Estimation] id={data.id} | Persons Detected={len(keypoints)} | "
            f"Timing: preprocess={t1 - start:.3f}s, inference={t2 - t1:.3f}s, postprocess={t3 - t2:.3f}s" 
        )

        return {
            "id": data.id,
            "count": len(keypoints),
            "keypoints": keypoints,
            "boxes": boxes,
            "speed_preprocess": round(t1 - start, 4),
            "speed_inference": round(t2 - t1, 4),
            "speed_postprocess": round(t3 - t2, 4)
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
