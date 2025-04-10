from pydantic import BaseModel

class PoseRequest(BaseModel):
    id: str
    image: str