from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import dlib
from personal_color_analysis import personal_color
import pandas as pd
import os
from face_shape_classify.align_face import align_face
from face_shape_classify.classify_face_shape import classify_face_shape
from face_shape_classify.preprocess_image import preprocess_image
from sentence_transformers import SentenceTransformer
from vector.feedback import search_glasses_with_feedback
from vector.milvus import (
    insert_data_to_milvus,
    query_milvus, extract_query,
)

app = Flask(__name__)

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def index():
    return render_template('index.html')


# 사진 업로드 및 얼굴 탐지 및 피부톤 추출, 얼굴형 분석, 추천 안경모델 검색
@app.route('/upload', methods=['POST'])
def upload():
    insert_data_to_milvus()
    files = request.files.getlist('image')
    if files:
        file = files[0]
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # img_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 이미지 저장 후 경로 전달
        img_path = './uploaded_image.jpg'
        cv2.imwrite(img_path, img_cv)

        # 디버깅: img_cv 확인
        print(f"img_cv shape: {img_cv.shape}")
        print(f"img_cv dtype: {img_cv.dtype}")

        # 피부톤 분석
        tone, skin_tone = personal_color.analysis(img_cv)

        # skin_tone을 int32에서 기본 int로 변환
        skin_tone = [int(c) for c in skin_tone]  # numpy int32 -> int 변환

        # 이미지 전처리
        img = preprocess_image(image_data)

        # 얼굴 감지
        faces = face_detector(img)
        if len(faces) == 0:
            return jsonify({"error": "No face detected"}), 400

        # 얼굴형 분류
        face_shapes = []
        for face in faces:
            landmarks = landmark_predictor(img, face)
            face_shape = classify_face_shape(landmarks)
            app.logger.info(f"Face shape detected: {face_shape}")

            if face_shape == "Unknown":
                app.logger.info("Face shape is unknown, aligning face...")
                aligned_img = align_face(img, face)
                if aligned_img is not None:
                    # 재정렬된 얼굴로 다시 분류 시도
                    landmarks = landmark_predictor(aligned_img, face)
                    face_shape = classify_face_shape(landmarks)
                    app.logger.info(f"Face shape after alignment: {face_shape}")
                else:
                    app.logger.error("Face alignment failed")
            face_shapes.append(face_shape)

        # 결과 반환
        if face_shapes and "Unknown" not in face_shapes:
            print(face_shapes)
        else:
            return jsonify({"face_shape": "Unknown"}), 400

        face_shape = face_shapes[0]
        face_shape += "형"
        print("Face_shpae 첫번째 요소", face_shape)
        skin_tone2 = tone
        print("skin_tone=====", skin_tone2)

        extracted_query = extract_query(face_shape, skin_tone2)
        print(extracted_query)
        query_vector = model.encode([extracted_query])[0]

        # Milvus에서 검색
        results = query_milvus(query_vector)
        results = sorted(results, key=lambda result: result.distance)


        print(face_shape)
        # 결과 반환
        return jsonify({
            "tone": tone,
            "face_shape": face_shape,
            "glasses_id": [result.id for result in results]
        })
    else:
        return jsonify({"error": "No image uploaded"}), 400


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.json  # JSON 데이터를 가져옴
    query = data.get("feedback")
    face_shape = data.get("face_shape")
    skin_tone2 = data.get("personal_color")
    # query = request.args.get("feedback")
    # face_shape = request.args.get("face_shape")
    # skin_tone2 = request.args.get("personal_color")
    print("###feedback", query)
    print("#####face_shape", face_shape, skin_tone2)
    extracted_query = extract_query(face_shape, skin_tone2)
    query_vector = model.encode([extracted_query])[0]
    # Milvus에서 검색
    results = query_milvus(query_vector)
    results = sorted(results, key=lambda result: result.distance)

    results_ = search_glasses_with_feedback(query_vector, [result.text for result in results], query,
                                            "glasses_collection")

    # return jsonify([
    #     result.id for result in results_
    # ])
    return jsonify({
        "glasses_id": [result.id for result in results_]
    })


if __name__ == '__main__':
    app.run(debug=True)


