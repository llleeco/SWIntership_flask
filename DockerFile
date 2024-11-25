#  Python 이미지를 기반으로 사용 
FROM python:3.9-slim

#  작업 디렉토리 설정
WORKDIR /app

#  필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip

#  애플리케이션 파일 복사
COPY requirements.txt /app/

#  Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

#  애플리케이션 코드 복사
COPY . /app/

#  Flask 애플리케이션 실행 (기본 Flask 포트 5000)
EXPOSE 5000

#  컨테이너 시작 시 실행할 명령
CMD ["python", "app.py"]