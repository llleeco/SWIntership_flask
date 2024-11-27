import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np


# Chroma 클라이언트 초기화
chroma_client = chromadb.Client()
collection_name = 'eyewear_recommendations'

# Sentence-BERT 모델 로드
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def add_glasses_to_chroma(df, eyewear_collection):
    for index, row in df.iterrows():
        # 데이터 행을 딕셔너리 형태로 변환
        glasses_data = row.to_dict()

        # 임베딩 생성
        embedding = generate_embedding_from_data(glasses_data)

        # 벡터 DB에 추가
        eyewear_collection.add(
            ids=[str(glasses_data['id'])],  # 고유 ID
            documents=[create_combined_text(glasses_data)],  # 속성 설명 문자열
            embeddings=[embedding]  # 임베딩 데이터
        )


def create_combined_text(data):
    # 각 속성을 명확히 구분하여 텍스트 생성
    combined_text = (
        f"Model: {data['model']}, "
        f"Color: {data['color']}, "
        f"Shape: {data['shape']}, "
        f"Size: {data['width']}x{data['length']} mm, "
        f"Weight: {data['weight']} grams, "
        f"Material: {data['material']}, "
        f"Price: {data['price']}"
    )
    return combined_text


def generate_embedding_from_data(data):
    # 텍스트 생성
    combined_text = create_combined_text(data)

    # 결합된 텍스트로 임베딩 생성
    embedding = model.encode(combined_text).tolist()
    return embedding

# Chroma 컬렉션 생성 (이미 있는 경우 불러옴)
eyewear_collection = chroma_client.get_or_create_collection(name=collection_name)

# 예시 매핑 테이블
mapping_table = [
    {'face_shape': '둥근형', 'skin_tone': '웜톤', 'recommended_colors': ['brown', 'gold'], 'recommended_shapes': ['square', 'poly']},
    {'face_shape': '네모형', 'skin_tone': '쿨톤', 'recommended_colors': ['silver', 'navy'], 'recommended_shapes': ['round']},
]

def search_glasses_by_combined_mapping(user_face_shape, user_skin_tone, eyewear_collection):
    for entry in mapping_table:
        print("user_face_shape:", user_face_shape)
        print("user_face_shape:", user_skin_tone)
        print(entry)
        if entry['face_shape'] == user_face_shape and entry['skin_tone'] == user_skin_tone:
            # 색상과 모양을 하나의 문장으로 결합
            combined_attributes = f"Color: {', '.join(entry['recommended_colors'])}, Shape: {', '.join(entry['recommended_shapes'])}"
            print(combined_attributes)
            # 결합된 텍스트로 임베딩 생성
            combined_embedding = model.encode(combined_attributes).tolist()

            # ChromaDB에서 검색
            results = eyewear_collection.query(
                query_embeddings=[combined_embedding],
                n_results=5,
                include = ["documents", "distances"]
            )
            print("Search Results:", results)
            return results['documents']  # 검색된 안경 모델 리스트 반환

    # 매핑 테이블에서 일치하는 조건이 없는 경우 빈 리스트 반환
    return []

def search_glasses_by_or_condition(user_face_shape, user_skin_tone, eyewear_collection):
    for entry in mapping_table:
        if entry['face_shape'] == user_face_shape and entry['skin_tone'] == user_skin_tone:
            # 색상과 모양 조건 생성
            color_queries = [f"Color: {color}" for color in entry['recommended_colors']]
            shape_queries = [f"Shape: {shape}" for shape in entry['recommended_shapes']]

            # 색상 검색
            color_results = []
            for color_query in color_queries:
                embedding = model.encode(color_query).tolist()
                results = eyewear_collection.query(
                    query_embeddings=[embedding],
                    n_results=10,
                    include=["documents", "distances"]
                )
                # 쌍으로 저장하여 길이 불일치 방지
                color_results.extend(zip(results['documents'], results['distances'][0]))

            # 모양 검색
            shape_results = []
            for shape_query in shape_queries:
                embedding = model.encode(shape_query).tolist()
                results = eyewear_collection.query(
                    query_embeddings=[embedding],
                    n_results=10,
                    include=["documents", "distances"]
                )
                # 쌍으로 저장하여 길이 불일치 방지
                shape_results.extend(zip(results['documents'], results['distances'][0]))

            # 결과 결합 및 점수 계산
            all_results = {}
            for doc, score in color_results:
                doc_key = tuple(doc)  # 리스트를 튜플로 변환
                if doc_key not in all_results:
                    all_results[doc_key] = {'color_score': 1 - score, 'shape_score': 0}  # 색상 점수만 추가

            for doc, score in shape_results:
                doc_key = tuple(doc)  # 리스트를 튜플로 변환
                if doc_key not in all_results:
                    all_results[doc_key] = {'color_score': 0, 'shape_score': 1 - score}  # 모양 점수만 추가
                else:
                    all_results[doc_key]['shape_score'] = 1 - score  # 모양 점수 업데이트

            # 최종 점수 계산 (가중 평균)
            final_results = sorted(
                all_results.items(),
                key=lambda x: 0.5 * x[1]['color_score'] + 0.5 * x[1]['shape_score'],  # 가중치 0.5씩 부여
                reverse=True  # 점수가 높은 순서대로 정렬
            )

            # 정렬된 결과 반환
            return [list(result[0]) for result in final_results]  # 튜플을 리스트로 변환하여 반환

    # 조건에 맞는 결과가 없으면 빈 리스트 반환
    return []


# def search_glasses_with_equal_weights(user_face_shape, user_skin_tone, eyewear_collection):
#     for entry in mapping_table:
#         if entry['face_shape'] == user_face_shape and entry['skin_tone'] == user_skin_tone:
#             # 색상과 모양 속성 결합
#             # 쿼리에서 색상과 모양을 개별적으로 분리
#             color_queries = [f"Color: {color}" for color in entry['recommended_colors']]
#             shape_queries = [f"Shape: {shape}" for shape in entry['recommended_shapes']]
#
#             # 각각 임베딩 생성
#             color_embeddings = [model.encode(color_query).tolist() for color_query in color_queries]
#             shape_embeddings = [model.encode(shape_query).tolist() for shape_query in shape_queries]
#
#             # 모든 색상/모양 임베딩의 평균 계산
#             combined_color_embedding = [sum(vec) / len(vec) for vec in zip(*color_embeddings)]
#             combined_shape_embedding = [sum(vec) / len(vec) for vec in zip(*shape_embeddings)]
#
#             # 최종 결합 임베딩 (가중치 동일)
#             combined_embedding = [
#                 0.6 * c + 0.4 * s for c, s in zip(combined_color_embedding, combined_shape_embedding)
#             ]
#
#             # ChromaDB에서 검색
#             results = eyewear_collection.query(
#                 query_embeddings=[combined_embedding],
#                 n_results=10,
#                 include=["documents", "distances"]
#             )
#
#             # 유사도 기준 정렬
#             sorted_results = sorted(
#                 zip(results['documents'], results['distances']),
#                 key=lambda x: x[1]  # 거리 기준 오름차순
#             )
#
#             # 정렬된 문서 반환
#             return [doc for doc, _ in sorted_results]
#
#     # 매핑 테이블 조건에 맞는 항목이 없을 경우 빈 리스트 반환
#     return []
