import base64
import io
import numpy as np
from PIL import Image
import cv2
from fastapi import HTTPException

def base64_to_pil(base64_str: str) -> Image.Image:
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image input: {str(e)}")

def pil_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def bytes_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL image (RGB) to OpenCV format (BGR) with resizing and compression"""
    image = image.resize((224, 224))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    compressed_image = Image.open(buffer)

    return cv2.cvtColor(np.array(compressed_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image (BGR) to PIL format (RGB)"""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
