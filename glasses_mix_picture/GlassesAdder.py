import cv2
import dlib
import numpy as np

class GlassesAdder:
    def __init__(self, landmark_model_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(landmark_model_path)

    def add_glasses(self, face_img, glasses_img, offset_y=0):
        if face_img is None or glasses_img is None:
            raise ValueError("Input images must not be None!")

        # 안경 이미지의 배경 제거
        glasses_img = self.remove_background(glasses_img)

        # 얼굴과 랜드마크 감지
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            raise ValueError("No faces detected!")

        for face in faces:
            landmarks = self.predictor(gray, face)

            # 얼굴 너비 계산 (좌우 광대뼈 사이 거리)
            face_width = np.linalg.norm(
                np.array([landmarks.part(16).x, landmarks.part(16).y]) - 
                np.array([landmarks.part(0).x, landmarks.part(0).y])
            )

            # 안경 크기를 얼굴 너비 기준으로 조정
            glasses_width = int(face_width * 1.1)  # 얼굴 너비의 1.1배로 설정
            glasses_height = int(glasses_img.shape[0] * (glasses_width / glasses_img.shape[1]))
            resized_glasses = cv2.resize(glasses_img, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

            # 두 눈 중심 좌표 계산
            left_eye_center = np.mean(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],
                axis=0
            ).astype(int)
            right_eye_center = np.mean(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)],
                axis=0
            ).astype(int)

            # 두 눈 중심 계산
            eye_center = (
                (left_eye_center[0] + right_eye_center[0]) // 2,
                (left_eye_center[1] + right_eye_center[1]) // 2
            )

            # 안경의 중앙 좌표를 기준으로 위치 계산
            top_left_x = eye_center[0] - glasses_width // 2
            top_left_y = eye_center[1] - glasses_height // 2 + offset_y

            # 얼굴 이미지 위에 안경을 합성 (알파 채널로 처리)
            self.overlay_image_alpha(face_img, resized_glasses, top_left_x, top_left_y)

        return face_img

    def remove_background(self, image):
        """이미지의 배경을 투명하게 만든 후 제거합니다."""
        if image.shape[2] == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 200], dtype=np.uint8)
            upper_white = np.array([180, 30, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_white, upper_white)

            mask_inv = cv2.bitwise_not(mask)
            alpha = mask_inv

            b, g, r = cv2.split(image)
            image = cv2.merge([b, g, r, alpha])
            return image

        elif image.shape[2] == 4:
            alpha = image[:, :, 3]
            if np.all(alpha == 255):
                bgr = image[:, :, :3]
                return self.remove_background(bgr)
            else:
                return image
        else:
            raise ValueError("Unsupported image format.")

    @staticmethod
    def overlay_image_alpha(background, overlay, x_offset, y_offset):
        overlay_h, overlay_w = overlay.shape[:2]

        # 합성 영역 조정
        if y_offset < 0:
            overlay = overlay[-y_offset:, :, :]
            overlay_h = overlay.shape[0]
            y_offset = 0
        if x_offset < 0:
            overlay = overlay[:, -x_offset:, :]
            overlay_w = overlay.shape[1]
            x_offset = 0
        if y_offset + overlay_h > background.shape[0]:
            overlay_h = background.shape[0] - y_offset
            overlay = overlay[:overlay_h, :, :]
        if x_offset + overlay_w > background.shape[1]:
            overlay_w = background.shape[1] - x_offset
            overlay = overlay[:, :overlay_w, :]

        overlay_rgb = overlay[..., :3]
        overlay_alpha = overlay[..., 3:] / 255.0

        background_roi = background[y_offset:y_offset + overlay_h, x_offset:x_offset + overlay_w]

        background_roi[:] = overlay_alpha * overlay_rgb + (1 - overlay_alpha) * background_roi