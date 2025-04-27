import base64
import requests

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_pose_estimation(base64_img: str):
    payload = {
        "id": "test-001",
        "image": base64_img
    }
    url = "http://localhost:60001/api/pose_estimation"
    response = requests.post(url, json=payload)
    print("pose_estimation response: ")
    print("Status Code:", response.status_code)
    print("Response:", response.json())

def test_pose_estimation_annotation(base64_img: str):
    payload = {
        "id": "test-002",
        "image": base64_img
    }
    url = "http://localhost:60001/api/pose_estimation_annotation"
    response = requests.post(url, json=payload)
    print("pose_estimation_annotation response:")
    print("Status Code:", response.status_code)

    if response.status_code == 200:
        res = response.json()

        image_bytes = base64.b64decode(res["image"])
        with open("result_annotated.jpg", "wb") as f:
            f.write(image_bytes)
        print("save result_annotated.jpg")
    else:
        print(" failed")

if __name__ == "__main__":
    image_base64 = encode_image("000000393149.jpg")
    test_pose_estimation(image_base64)
    test_pose_estimation_annotation(image_base64)
