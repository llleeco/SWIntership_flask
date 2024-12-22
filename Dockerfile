# Python 3.11 이미지를 기반으로 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 패키지 사전 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# 종속성 사전 설치 (numpy, cmake)
RUN pip install --no-cache-dir numpy cmake

# dlib 개별 설치 (캐싱 활용)
RUN pip install --no-cache-dir dlib

# Python 종속성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app/

# Flask 실행 관련 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Flask 실행을 위한 포트 노출
EXPOSE ${FLASK_RUN_PORT}

# 컨테이너 시작 시 Flask 실행
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]