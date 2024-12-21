# Python 3.11 이미지를 기반으로 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요한 OS 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# Python 종속성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . /app/

# Flask 애플리케이션 실행을 위한 포트 노출
EXPOSE 5000

# 환경 변수 설정 (Flask 실행)
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 컨테이너 시작 시 Flask 실행
CMD ["flask", "run"]