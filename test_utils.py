from app.utils.image import base64_to_pil, pil_to_base64, pil_to_cv2, cv2_to_pil
from PIL import Image
import cv2
import base64

# 1. reading original image
original = Image.open("test.jpg").convert("RGB")

# 2. PIL to base64
base64_str = pil_to_base64(original)
print("true, pil_to_base64 successful")

# 3. base64 to PIL
pil_img = base64_to_pil(base64_str)
print("true, base64_to_pil successful")

# 4. PIL to OpenCV
cv2_img = pil_to_cv2(pil_img)
print("true, pil_to_cv2 successful, Image shape:", cv2_img.shape)

# 5. OpenCV to PIL
pil_restored = cv2_to_pil(cv2_img)
print("true, cv2_to_pil successful")

# 6. save, check if color is correct
original.save("check_1_original.jpg")
pil_img.save("check_2_pil_decoded.jpg")
pil_restored.save("check_3_pil_from_cv2.jpg")
cv2.imwrite("check_4_cv2_img.jpg", cv2_img)

print("true, all successful")
