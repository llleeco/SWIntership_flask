from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from PIL import Image
import io
from sentence_transformers import SentenceTransformer

from vector.feedback import search_glasses_with_feedback
from vector.milvus5 import (
    insert_data_to_milvus,
    query_milvus, extract_query
)

app = Flask(__name__)

#웹 페이지 렌더링
@app.route('/')
def index():
    return render_template('index.html')

# 사진 업로드 및 얼굴 탐지 및 피부톤 추출
@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('image')
    if files:
        file = files[0]
        image_data = file.read()
        image = Image.open(io.BytesIO(image_data))

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
  
        # 이미지 저장 후 경로 전달
        img_path = './uploaded_image.jpg'
        cv2.imwrite(img_path, img_cv)
        # 피부톤 분석
        tone, skin_tone = personal_color.analysis(img_cv)
        
        # skin_tone을 int32에서 기본 int로 변환
        skin_tone = [int(c) for c in skin_tone]  # numpy int32 -> int 변환
        
        return jsonify({
            "tone": tone,
            "skin_tone": skin_tone
        })
    else:
        return jsonify({"error": "No image uploaded"}), 400

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
@app.route("/search", methods=["POST"])
def search():
    insert_data_to_milvus()
    face_shape = request.args.get("face_shape")
    skin_tone = request.args.get("skin_tone")

    extracted_query = extract_query(face_shape, skin_tone)
    print(extracted_query)
    query_vector = model.encode([extracted_query])[0]
    # Milvus에서 검색
    results = query_milvus(query_vector)
    results = sorted(results, key=lambda result: result.distance)
    # 결과 반환
    return jsonify([
        result.id for result in results
    ])

@app.route("/feedback", methods=["POST"])
def feedback():
    insert_data_to_milvus()
    query = request.args.get("query")
    extracted_query = extract_query("둥근형", "가을웜톤")
    print(extracted_query)
    query_vector = model.encode([extracted_query])[0]
    # Milvus에서 검색
    results = query_milvus(query_vector)
    print(results)
    results = sorted(results, key=lambda result: result.distance)
    results = search_glasses_with_feedback(query_vector, [result.text for result in results], query, "glasses_collection")

    return jsonify([
        result.id for result in results
    ])






if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run('0.0.0.0',debug=True)
