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

# pip 업그레이드 및 캐시 제거
RUN pip install --upgrade pip && pip cache purge

# Python 종속성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사 (변경될 가능성이 높음)
COPY . /app/

# Flask 실행 관련 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=production

# Flask 실행을 위한 포트 노출
EXPOSE ${FLASK_RUN_PORT}

# 헬스체크 (컨테이너 상태 확인)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${FLASK_RUN_PORT}/ || exit 1

# 컨테이너 시작 시 Flask 실행
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]