from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import base64
from PIL import Image
import io
import time
#from personal_color_analysis import personal_color
from vector.chroma import search_chroma, get_suggested_glasses, filter_glasses_by_mapping, \
    vector_search_in_chroma, eyewear_collection, add_glasses_to_chroma, search_glasses_by_combined_mapping, \
    search_glasses_with_equal_weights
import pandas as pd
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


@app.route('/add_glasses', methods=['POST'])
def add_glasses():
    # 액셀 파일 경로로부터 데이터를 읽기
    excel_file_path = '안경.xlsx'
    df = pd.read_excel(excel_file_path)

    # Chroma 컬렉션에 안경 데이터 추가
    add_glasses_to_chroma(df, eyewear_collection)

    return jsonify({"message": "Glasses added successfully!"})
@app.route('/search', methods=['GET'])
def search_item():
    # face_shape = request.args.get('face_shape')
    # skin_tone = request.args.get('skin_tone')
    user_face_shape = '둥근형'
    user_skin_tone = '웜톤'
    # suggested_glasses = get_suggested_glasses(user_face_shape, user_skin_tone)
    # filtered_glasses = filter_glasses_by_mapping(suggested_glasses, eyewear_collection)
    # final_recommendations = vector_search_in_chroma(filtered_glasses, eyewear_collection)
    # # results = search_chroma(query)
    final_recommendations = search_glasses_with_equal_weights(user_face_shape, user_skin_tone, eyewear_collection)
    return jsonify(final_recommendations)

if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    app.run('0.0.0.0',debug=True)
