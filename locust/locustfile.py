# locustfile.py

from locust import HttpUser, task, between
import base64
import random

class PoseEstimationUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        with open("sample.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            self.image_data = encoded_string

    @task
    def send_pose_request(self):
        payload = {
            "id": f"test_{random.randint(1,10000)}",
            "img": self.image_data
        }
        headers = {"Content-Type": "application/json"}
        self.client.post("/api/pose_estimation_annotation", json=payload, headers=headers)
