import time
import traceback
import logging
from fastapi import HTTPException
from app.utils.image import base64_to_pil, pil_to_cv2, bytes_to_base64
from models.detector import Detector

logger = logging.getLogger(__name__)

class PoseService:
    def __init__(self, model_path: str = "models/yolo11l-pose.pt"):
        self.detector = Detector()

    def detect(self, data):
        try:
            logger.info(f"[Detect] Start processing id={data.id}")
            start_time = time.time()

            pil_img = base64_to_pil(data.image)
            cv2_img = pil_to_cv2(pil_img)
            preprocess_time = time.time()

            keypoints, boxes = self.detector.predict(cv2_img)
            inference_time = time.time()

            if keypoints is None or boxes is None:
                logger.warning(f"[Detect] id={data.id} | keypoints or boxes is None (model may have failed). Returning empty list.")
                keypoints, boxes = [], []

            if not isinstance(keypoints, list) or not isinstance(boxes, list):
                logger.warning(f"[Detect] id={data.id} | keypoints or boxes not list after detection. Forcing to empty list.")
                keypoints, boxes = [], []

            time.sleep(0.01)
            postprocess_time = time.time()

            logger.info(
                f"[Pose Estimation] id={data.id} | Persons Detected={len(keypoints)} | "
                f"Timing: preprocess={preprocess_time - start_time:.3f}s, "
                f"inference={inference_time - preprocess_time:.3f}s, "
                f"postprocess={postprocess_time - inference_time:.3f}s"
            )

            return {
                "id": data.id,
                "count": len(keypoints),
                "keypoints": keypoints,
                "boxes": boxes,
                "speed_preprocess": round(preprocess_time - start_time, 4),
                "speed_inference": round(inference_time - preprocess_time, 4),
                "speed_postprocess": round(postprocess_time - inference_time, 4)
            }

        except Exception as e:
            logger.error(f"[Detect][Error] id={getattr(data, 'id', 'unknown')} | {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal error during pose detection.")

    def detect_with_annotation(self, data):
        try:
            logger.info(f"[Annotate] Start annotation for id={data.id}")

            pil_img = base64_to_pil(data.image)
            cv2_img = pil_to_cv2(pil_img)

            annotated_bytes = self.detector.predict_with_annotation(cv2_img)

            if not annotated_bytes:
                logger.warning(f"[Annotate] id={data.id} | No annotated image generated.")
                return {
                    "id": data.id,
                    "image": None,
                    "message": "No annotated image generated because no keypoints detected."
                }

            result_base64 = bytes_to_base64(annotated_bytes)

            logger.info(f"[Annotate] Finished annotation for id={data.id}")
            return {
                "id": data.id,
                "image": result_base64
            }

        except Exception as e:
            logger.error(f"[Annotate][Error] id={getattr(data, 'id', 'unknown')} | {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal error during annotation.")
