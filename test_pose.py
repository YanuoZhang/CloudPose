import requests
import base64


API_URL = "http://localhost:60001/api/pose_estimation_annotation"

IMAGE_PATH = "test.jpg"  

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def main():
    base64_image = encode_image_to_base64(IMAGE_PATH)
    payload = {
        "id": "test123",
        "image": base64_image
    }

    try:
        response = requests.post(API_URL, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response JSON:")
        print(response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    main()
