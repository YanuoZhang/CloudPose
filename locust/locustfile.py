# locustfile.py

from locust import HttpUser, task, between
import base64
import uuid
import json
import os
import random

class CloudPoseUser(HttpUser):
    wait_time = between(0.5, 1) 

    def on_start(self):
        input_folder = "./inputfolder/"
        self.images = []
        for filename in os.listdir(input_folder):
            if filename.lower().endswith(".jpg"):
                path = os.path.join(input_folder, filename)
                self.images.append(path)

    def generate_payload(self):
        image_path = random.choice(self.images)
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        payload = {
            "id": str(uuid.uuid4()),
            "image": encoded_string
        }
        return payload

    @task(1)
    def pose_estimation(self):
        payload = self.generate_payload()
        headers = {'Content-Type': 'application/json'}
        self.client.post("/api/pose_estimation", data=json.dumps(payload), headers=headers)

    @task(1)
    def pose_estimation_annotation(self):
        payload = self.generate_payload()
        headers = {'Content-Type': 'application/json'}
        self.client.post("/api/pose_estimation_annotation", data=json.dumps(payload), headers=headers)
