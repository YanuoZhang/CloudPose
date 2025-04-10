import time

class PoseService:
    def detect(self, data):
        return {
            "id": data.id,
            "count": 1,
            "boxes": [{"x": 100, "y": 100, "width": 50, "height": 50, "probability": 0.95}],
            "keypoints": [[[100, 100, 0.9] for _ in range(17)]],
            "speed_preprocess": 0.01,
            "speed_inference": 0.05,
            "speed_postprocess": 0.01
        }

    def detect_with_annotation(self, data):
        return {
            "id": data.id,
            "image": "fake_base64_encoded_image"
        }
