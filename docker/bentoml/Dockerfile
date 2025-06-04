FROM python:3.10-slim

WORKDIR /app

# 시스템 필수 패키지 업데이트 및 pip 업그레이드
RUN apt-get update && \
    apt-get install -y gcc && \
    pip install --upgrade pip

# 필수 Python 패키지 설치
RUN pip install --no-cache-dir \
    scikit-learn \
    pandas \
    mlflow \
    bentoml

# 파일 복사
COPY ./bento_service /app/bento_service
COPY ./scripts/train.py /app/train.py


# API 서빙 실행
CMD ["bentoml", "serve", "bento_service:svc"]


# bento_service.py 복사
COPY ./bento_service.py /app/bento_service.py

# BentoML 실행 시 서비스 위치 지정
ENV BENTOML_SERVE=bento_service:svc
