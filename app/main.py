from fastapi import FastAPI
from app.schemas.request import PoseRequest
from app.services.pose_service import PoseService

app = FastAPI()
pose_service = PoseService()

@app.post("/api/pose_estimation")
def pose_estimation(data: PoseRequest):
    return pose_service.detect(data)

@app.post("/api/pose_estimation_annotation")
def pose_estimation_annotation(data: PoseRequest):
    return pose_service.detect_with_annotation(data)