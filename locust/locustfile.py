# Optimized locustfile.py for CloudPose service

from locust import HttpUser, task, between, events
import base64
import uuid
import json
import os
import random
import sys

first_failure_detected = False 

class CloudPoseUser(HttpUser):
    wait_time = between(1, 5)

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

    @task
    def pose_estimation(self):
        payload = self.generate_payload()
        headers = {'Content-Type': 'application/json'}
        with self.client.post("/api/pose_estimation"
                               , data=json.dumps(payload)
                               , headers=headers
                               , catch_response=True) as response:
            if response.status_code == 200:
                if response.elapsed.total_seconds() > 30:
                    response.failure(f"Response time {response.elapsed.total_seconds()}s exceeds {30}s threshold")
                else:
                    response.success()
            else:
                response.failure(f"pose_estimation failed with status {response.status_code}")


    @task
    def pose_estimation_annotation(self):
        payload = self.generate_payload()
        headers = {'Content-Type': 'application/json'}
        with self.client.post("/api/pose_estimation_annotation"
                              , data=json.dumps(payload)
                              , headers=headers
                              , catch_response=True) as response:
            if response.status_code == 200:
                if response.elapsed.total_seconds() > 30:
                    response.failure(f"Response time {response.elapsed.total_seconds()}s exceeds {30}s threshold")
                else:
                    response.success()
            else:
                response.failure(f"pose_estimation failed with status {response.status_code}")
    
    def send_request(self, path):
        payload = self.generate_payload()
        headers = {'Content-Type': 'application/json'}
        with self.client.post(
            path, 
            data=json.dumps(payload), 
            headers=headers,
            catch_response=True,
            timeout=120
        ) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status {response.status_code}")

@events.request_failure.add_listener
def on_failure(request_type, name, response_time, exception, **kwargs):
    global first_failure_detected
    if not first_failure_detected:
        first_failure_detected = True
        print(" First failure detected!")
        print(f"User Count at Failure: {kwargs.get('user_count', 'unknown')}")
        sys.exit(0)