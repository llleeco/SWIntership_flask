import cv2
import numpy as np
import dlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def classify_face_shape(landmarks):
    # 턱, 얼굴 높이/너비 측정
    jaw_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(0, 17)])
    face_height = np.linalg.norm(
        np.array([landmarks.part(8).x, landmarks.part(8).y]) - np.array([landmarks.part(27).x, landmarks.part(27).y])
    )
    face_width = np.linalg.norm(
        np.array([landmarks.part(0).x, landmarks.part(0).y]) - np.array([landmarks.part(16).x, landmarks.part(16).y])
    )
    jaw_width = np.linalg.norm(
        np.array([landmarks.part(4).x, landmarks.part(4).y]) - np.array([landmarks.part(12).x, landmarks.part(12).y])
    )

    jaw_ratio = jaw_width / face_width
    face_ratio = face_width / face_height * 1.5

    # 얼굴형 데이터
    face_shapes = {
        "하트": {"jaw_ratio": [0.74, 0.83], "face_ratio": [1.68, 1.99]},
        "긴": {"jaw_ratio": [0.74, 0.83], "face_ratio": [1.78, 2.05]},
        "둥근": {"jaw_ratio": [0.77, 0.86], "face_ratio": [1.69, 1.96]},
        "각진": {"jaw_ratio": [0.79, 0.88], "face_ratio": [1.73, 1.98]},
        "타원": {"jaw_ratio": [0.73, 0.84], "face_ratio": [1.76, 2.02]},
        "다이아": {"jaw_ratio": [0.74, 0.83], "face_ratio": [1.67, 1.94]},
    }

    # 가장 유사한 얼굴형 찾기
    min_distance = float('inf')
    best_match = None
    for shape, data in face_shapes.items():
        dist = abs(jaw_ratio - np.mean(data["jaw_ratio"])) + \
               abs(face_ratio - np.mean(data["face_ratio"]))
        if dist < min_distance:
            min_distance = dist
            best_match = shape

    print(f"입력 값: 턱 비율={jaw_ratio:.2f}, 광대 비율={face_ratio:.2f}")
    print(f"예측된 얼굴형: {best_match}")
    return best_match
