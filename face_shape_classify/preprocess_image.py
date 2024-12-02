import cv2
import numpy as np
import dlib
#import pillow
from PIL import Image as PILImage
import io

# 이미지 전처리 함수 (조명 개선)
def preprocess_image(image_data):
    try:
        """# image_file이 이미 PIL Image 객체일 때
        if isinstance(image_file, PILImage.Image):
            img = image_file
        else:
        """
    # 바이너리 데이터로 이미지 읽기
        img = PILImage.open(io.BytesIO(image_data)).convert("RGB")
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        return gray
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None