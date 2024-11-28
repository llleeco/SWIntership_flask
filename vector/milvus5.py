from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus.orm import utility
from sentence_transformers import SentenceTransformer
import pandas as pd

# Milvus 설정
COLLECTION_NAME = "glasses_collection"

# Milvus 연결
connections.connect("default", host="127.0.0.1", port="19530")

# Sentence-BERT 모델 로드
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def init_milvus():
    """Milvus 컬렉션 초기화"""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="brand", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="color", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="shape", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="width", dtype=DataType.DOUBLE),
        FieldSchema(name="length", dtype=DataType.DOUBLE),
        FieldSchema(name="weight", dtype=DataType.DOUBLE),
        FieldSchema(name="material", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="price", dtype=DataType.INT64),
        # Sentence-BERT 임베딩 크기
    ]

    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(name=COLLECTION_NAME)
        collection.drop()  # 기존 데이터 삭제
    schema = CollectionSchema(fields, description="Glasses vector collection")
    collection = Collection(COLLECTION_NAME, schema)
    return collection

def insert_data_to_milvus(data_path="안경.xlsx"):
    """엑셀 데이터 읽어서 벡터화 후 Milvus에 삽입"""
    # 데이터 로드
    df = pd.read_excel(data_path)

    # 각 데이터의 특징을 결합하여 임베딩 생성
    descriptions = df.apply(
        lambda row: f"Color: {row['color']}, Shape: {row['shape']}, Material: {row['material']}, Price: {row['price']}, Brand: {row['brand']}, Width: {row['width']}, Length: {row['length']}, Weight:{row['weight']}",
        axis=1
    )
    vectors = model.encode(descriptions.tolist())
    # Milvus에 삽입
    collection = init_milvus()
    collection.insert([df['id'].tolist(), vectors,descriptions.tolist(), df['brand'].tolist(),df['color'].tolist(),df['shape'].tolist(),df['width'].tolist(),df['length'].tolist(),df['weight'].tolist(),df['material'].tolist(),df['price'].tolist()])
    collection.create_index(field_name="vector", index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}})
    collection.load()


# Sentence-BERT 모델 로드

def query_milvus(query_vector, top_k=1):
    """Milvus에서 벡터 유사도 검색"""
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 벡터 검색
    results = collection.search(
        data=[query_vector],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        limit=top_k,
        output_fields=["id", "text", "color", "price", "width", "length", "weight", "material", "brand"]
    )

    return results[0]

# LLM으로 사용자 요청 처리
#openai.api_key = ""
    # llm_response = openai.chat.completions.create(
    #     model="gpt-3.5-turbo",
    #     messages=[
    #         {"role": "system",
    #          "content": (
    #              "You are an assistant that reformulates user queries about glasses into concise, clear text "
    #              "that includes essential details for database search. "
    #              "Base your response on database fields: 'color', 'shape', 'material', 'price', and 'weight'. "
    #              "Your task is to generate a single sentence that can describe a search query precisely."
    #              "Here is a mapping table and available options for recommendations: "
    #              f"{mapping_data}. Use this data to generate a single sentence. "
    #          )},
    #         {"role": "user",
    #          "content": f"Based on this query: '{user_query}', provide a refined search description."}
    #     ]
    # )
    # extracted_query = llm_response.choices[0].message.content

mapping_table = [
{'face_shape': '둥근형', 'skin_tone': '봄웜톤', 'recommended_colors': ['gold', 'brown', 'rosgold'], 'recommended_shapes': ['square', 'poly'], 'recommended_materials': ['metal', 'plastic']},
{'face_shape': '둥근형', 'skin_tone': '여름쿨톤', 'recommended_colors': ['silver', 'purple', 'navy'], 'recommended_shapes': ['square', 'frameless'], 'recommended_materials': ['titan', 'metal']},
{'face_shape': '둥근형', 'skin_tone': '가을웜톤', 'recommended_colors': ['brown', 'yellow', 'gold'], 'recommended_shapes': ['square', 'boeing'], 'recommended_materials': ['plastic', 'metal']},
{'face_shape': '둥근형', 'skin_tone': '겨울쿨톤', 'recommended_colors': ['black', 'silver', 'navy'], 'recommended_shapes': ['square', 'poly'], 'recommended_materials': ['titan', 'metal']},


{'face_shape': '각진형', 'skin_tone': '봄웜톤', 'recommended_colors': ['gold', 'yellow', 'brown'], 'recommended_shapes': ['round', 'orval'], 'recommended_materials': ['metal', 'plastic']},
{'face_shape': '각진형', 'skin_tone': '여름쿨톤', 'recommended_colors': ['silver', 'purple', 'transperant'], 'recommended_shapes': ['round', 'frameless'], 'recommended_materials': ['titan', 'metal']},
{'face_shape': '각진형', 'skin_tone': '가을웜톤', 'recommended_colors': ['brown', 'rosgold', 'gold'], 'recommended_shapes': ['round', 'cats'], 'recommended_materials': ['plastic', 'metal']},
{'face_shape': '각진형', 'skin_tone': '겨울쿨톤', 'recommended_colors': ['black', 'navy', 'silver'], 'recommended_shapes': ['round', 'boeing'], 'recommended_materials': ['titan', 'metal']},


{'face_shape': '타원형', 'skin_tone': '봄웜톤', 'recommended_colors': ['yellow', 'rosgold', 'gold'], 'recommended_shapes': ['cats', 'boeing'], 'recommended_materials': ['metal', 'plastic']},
{'face_shape': '타원형', 'skin_tone': '여름쿨톤', 'recommended_colors': ['silver', 'purple', 'transperant'], 'recommended_shapes': ['round', 'orval'], 'recommended_materials': ['titan', 'metal']},
{'face_shape': '타원형', 'skin_tone': '가을웜톤', 'recommended_colors': ['brown', 'gold', 'yellow'], 'recommended_shapes': ['cats', 'frameless'], 'recommended_materials': ['plastic', 'metal']},
{'face_shape': '타원형', 'skin_tone': '겨울쿨톤', 'recommended_colors': ['black', 'silver', 'navy'], 'recommended_shapes': ['orval', 'round'], 'recommended_materials': ['titan', 'metal']},


{'face_shape': '하트형', 'skin_tone': '봄웜톤', 'recommended_colors': ['rosgold', 'brown', 'yellow'], 'recommended_shapes': ['cats', 'poly'], 'recommended_materials': ['plastic', 'metal']},
{'face_shape': '하트형', 'skin_tone': '여름쿨톤', 'recommended_colors': ['transperant', 'purple', 'silver'], 'recommended_shapes': ['frameless', 'round'], 'recommended_materials': ['titan', 'metal']},
{'face_shape': '하트형', 'skin_tone': '가을웜톤', 'recommended_colors': ['gold', 'brown', 'rosgold'], 'recommended_shapes': ['boeing', 'orval'], 'recommended_materials': ['metal', 'plastic']},
{'face_shape': '하트형', 'skin_tone': '겨울쿨톤', 'recommended_colors': ['black', 'navy', 'silver'], 'recommended_shapes': ['cats', 'frameless'], 'recommended_materials': ['titan', 'metal']},


{'face_shape': '긴형', 'skin_tone': '봄웜톤', 'recommended_colors': ['gold', 'yellow', 'brown'], 'recommended_shapes': ['cats', 'round'], 'recommended_materials': ['metal', 'plastic']},
{'face_shape': '긴형', 'skin_tone': '여름쿨톤', 'recommended_colors': ['purple', 'silver', 'transperant'], 'recommended_shapes': ['frameless', 'boeing'], 'recommended_materials': ['titan', 'metal']},
{'face_shape': '긴형', 'skin_tone': '가을웜톤', 'recommended_colors': ['brown', 'rosgold', 'gold'], 'recommended_shapes': ['cats', 'orval'], 'recommended_materials': ['plastic', 'metal']},
{'face_shape': '긴형', 'skin_tone': '겨울쿨톤', 'recommended_colors': ['navy', 'black', 'silver'], 'recommended_shapes': ['round', 'frameless'], 'recommended_materials': ['titan', 'metal']},


{'face_shape': '다이아형', 'skin_tone': '봄웜톤', 'recommended_colors': ['yellow', 'gold', 'rosgold'], 'recommended_shapes': ['cats', 'boeing'], 'recommended_materials': ['plastic', 'metal']},
{'face_shape': '다이아형', 'skin_tone': '여름쿨톤', 'recommended_colors': ['silver', 'transperant', 'purple'], 'recommended_shapes': ['round', 'frameless'], 'recommended_materials': ['titan', 'metal']},
{'face_shape': '다이아형', 'skin_tone': '가을웜톤', 'recommended_colors': ['brown', 'gold', 'rosgold'], 'recommended_shapes': ['boeing', 'orval'], 'recommended_materials': ['metal', 'plastic']},
{'face_shape': '다이아형', 'skin_tone': '겨울쿨톤', 'recommended_colors': ['black', 'navy', 'silver'], 'recommended_shapes': ['frameless', 'poly'], 'recommended_materials': ['titan', 'metal']}
]

def extract_query(face_shape, skin_tone):
    recommended_colors = []
    recommended_shapes = []
    recommended_materials = []

    # face_shape와 skin_tone을 기반으로 추천 정보를 가져옴
    for entry in mapping_table:
        if entry["face_shape"] == face_shape and entry["skin_tone"] == skin_tone:
            recommended_colors = entry["recommended_colors"]
            recommended_shapes = entry["recommended_shapes"]
            recommended_materials = entry["recommended_materials"]
            break  # 조건에 맞는 첫 번째 항목을 찾으면 반복 종료

    # 빈 리스트가 아니라면 조건 문자열 생성
    color_conditions = ' '.join(
        [f'{color}' for color in recommended_colors]) if recommended_colors else ""
    shape_conditions = ' '.join(
        [f'{shape}' for shape in recommended_shapes]) if recommended_shapes else ""
    material_conditions = ' '.join(
        [f'{material}' for material in recommended_materials]) if recommended_materials else ""

    conditions = [color_conditions, shape_conditions, material_conditions]
    expr = ' '.join([f'{condition}' for condition in conditions])
    return expr