import cv2
import numpy as np
import dlib
from .preprocess_image import preprocess_image
import os


dir_path = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
dat_path = os.path.join(dir_path , 'model', 'shape_predictor_68_face_landmarks.dat')

face_detector = dlib.get_frontal_face_detector()

# # 파일이 존재하는지 확인
# if not os.path.isfile(dat_path):
#     raise FileNotFoundError(f"File not found at: {dat_path}")

landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 얼굴 회전 보정 함수
def align_face(image_path):
    image = preprocess_image(image_path) # image 변수에 preprocess_image 함수의 결과값 할당
    faces = face_detector(image)

    if len(faces) == 0:
        print("현재 카메라로 얼굴을 감지할 수 없습니다. 재시도해주세요.\nNo face detected")
        return None
    
    aligned_faces = []
    for face in faces:
        landmarks = landmark_predictor(image, face)

        # 눈 좌표 가져오기 (왼쪽 눈: 36~41, 오른쪽 눈: 42~47)
        left_eye = np.mean([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], axis=0)
        right_eye = np.mean([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], axis=0)

        # 각도 계산
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))  # 회전 각도 계산

        # 이미지 중앙을 기준으로 회전 행렬 생성
        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        #return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        # 회전된 이미지 생성
        rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        aligned_faces.append(rotated_image)  # 회전된 얼굴 이미지 저장

    # 여러 얼굴이 있을 경우, 첫 번째 얼굴을 반환
    return aligned_faces[0] if aligned_faces else None
