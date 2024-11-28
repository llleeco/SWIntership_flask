import openai
from pymilvus import Collection

def search_glasses_with_feedback(initial_vector, initial_recommendations, query, collection_name):
    """
    사용자의 피드백을 기반으로 Milvus에서 검색을 수행하는 함수.

    Parameters:
        query (str): 사용자 피드백 문장 (자연어).
        collection_name (str): Milvus 컬렉션 이름.

    Returns:
        list: 검색 결과 리스트 (ID, 브랜드, 가격, 무게, 재질 포함).
    """
    # OpenAI 모델을 사용하여 Milvus 쿼리 생성
    print("#############", initial_recommendations)
    llm_response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": (
                 "You are an assistant that converts user feedback into Milvus query expressions. "
                 "Milvus supports the following fields for filtering glasses: 'brand', 'color', 'shape', "
                 "'width', 'length', 'material', 'price', 'weight'. \n\n"
                 "Valid operators are: \n"
                 "- Equality: `==`, `!=`\n"
                 "- Comparison: `<`, `<=`, `>`, `>=`\n"
                 "- Logical Operators: `&&`, `||`\n\n"
                 "Examples of Milvus query expressions: \n"
                 "- `brand != \"BrandA\" && weight <= 150`\n"
                 "- `color == \"brown\" && shape == \"round\"`\n\n"
                 "Your task is to analyze user feedback and generate a valid Milvus query expression "
                 "that can be used to search the glasses database. Only return the expression, no extra details."
                 "Consider the initial recommendations and user feedback to refine the search."
             )},
            {"role": "user",
             "content": f"Initial recommendations: {initial_recommendations}. User feedback: {query}."}
        ]
    )

    # OpenAI로부터 Milvus 쿼리 표현식 받기
    milvus_expr = llm_response.choices[0].message.content.strip()
    print(milvus_expr)
    # Milvus 컬렉션 연결
    collection = Collection(collection_name)
    collection.load()
    # Milvus에서 검색 실행
    results = collection.search(
        data=[initial_vector],
        anns_field="vector",
        limit=3,
        param={"metric_type": "L2", "params": {"nprobe": 16}},
        expr=milvus_expr,
        output_fields=["price"]
    )
    print("@@@@@@@@@@@@", results)
    return results[0]