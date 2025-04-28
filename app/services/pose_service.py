import time
import traceback
import logging
from fastapi import HTTPException
from app.utils.image import base64_to_pil, pil_to_cv2, bytes_to_base64
from models.detector import Detector
import concurrent.futures

logger = logging.getLogger(__name__)

class PoseService:
    def __init__(self, model_path: str = "models/yolo11l-pose.pt"):
        self.detector = Detector()
        self.cache = {}

    def detect(self, data):
        try:
            logger.info(f"[Detect] Start processing id={data.id}")

            if data.image in self.cache:
                logger.info(f"[Detect] Cache hit for id={data.id}")
                return self.cache[data.image]
            
            start_time = time.time()

            pil_img = base64_to_pil(data.image)
            cv2_img = pil_to_cv2(pil_img)
            preprocess_time = time.time()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.detector.predict, cv2_img)
                try:
                    keypoints, boxes = future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"[Detect][Timeout] id={data.id} | Detection timeout, returning empty result.")
                    keypoints, boxes = [], []

            inference_time = time.time()

            if keypoints is None or boxes is None:
                logger.warning(f"[Detect] id={data.id} | keypoints or boxes is None (model may have failed). Returning empty list.")
                keypoints, boxes = [], []

            if not isinstance(keypoints, list) or not isinstance(boxes, list):
                logger.warning(f"[Detect] id={data.id} | keypoints or boxes not list after detection. Forcing to empty list.")
                keypoints, boxes = [], []

            time.sleep(0.001)
            postprocess_time = time.time()

            logger.info(
                f"[Pose Estimation] id={data.id} | Persons Detected={len(keypoints)} | "
                f"Timing: preprocess={preprocess_time - start_time:.3f}s, "
                f"inference={inference_time - preprocess_time:.3f}s, "
                f"postprocess={postprocess_time - inference_time:.3f}s"
            )

            keypoints = keypoints if isinstance(keypoints, list) else []
            boxes = boxes if isinstance(boxes, list) else []

            result = {
                "id": data.id,
                "count": len(keypoints),
                "keypoints": keypoints,
                "boxes": boxes,
                "speed_preprocess": round(preprocess_time - start_time, 4),
                "speed_inference": round(inference_time - preprocess_time, 4),
                "speed_postprocess": round(postprocess_time - inference_time, 4)
            }

            self.cache[data.image] = result
            return result

        except Exception as e:
            logger.error(f"[Detect][Error] id={getattr(data, 'id', 'unknown')} | {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail="Internal error during pose detection.")


    def detect_with_annotation(self, data):
        try:
            logger.info(f"[Annotate] Start annotation for id={data.id}")

            pil_img = base64_to_pil(data.image)
            cv2_img = pil_to_cv2(pil_img)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.detector.predict_with_annotation, cv2_img)
                try:
                    annotated_bytes = future.result(timeout=5)
                except concurrent.futures.TimeoutError:
                    logger.warning(f"[Annotate][Timeout] id={data.id} | Annotation timeout, returning No annotated image.")
                    annotated_bytes = None

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
