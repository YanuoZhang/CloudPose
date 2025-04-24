from fastapi import FastAPI
from app.schemas.request import PoseRequest
from app.services.pose_service import PoseService
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CloudPose", description="Pose Detection API", version="1.0.0")
pose_service = PoseService()

@app.post("/api/pose_estimation")
def pose_estimation(data: PoseRequest):
    logger.info(f"Received pose_estimation request with id={data.id}")
    return pose_service.detect(data)

@app.post("/api/pose_estimation_annotation")
def pose_estimation_annotation(data: PoseRequest):
    logger.info(f"Received pose_estimation_annotation request with id={data.id}")
    return pose_service.detect_with_annotation(data)