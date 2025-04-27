from fastapi import FastAPI, Request
from app.schemas.request import PoseRequest
from app.services.pose_service import PoseService
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CloudPose", description="Pose Detection API", version="1.0.0")
pose_service = PoseService()

@app.post("/api/pose_estimation")
async def pose_estimation(req: Request):
    body = await req.json()
    if isinstance(body, str):
        body = json.loads(body)
    data = PoseRequest(**body)

    logger.info(f"Received pose_estimation request with id={data.id}")
    return pose_service.detect(data)

@app.post("/api/pose_estimation_annotation")
async def pose_estimation_annotation(req: Request):
    body = await req.json()
    if isinstance(body, str):
        body = json.loads(body)
    data = PoseRequest(**body)

    logger.info(f"Received pose_estimation_annotation request with id={data.id}")
    return pose_service.detect_with_annotation(data)
