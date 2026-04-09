FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
       --extra-index-url https://download.pytorch.org/whl/cu124

COPY app/ ./app/

# HuggingFace 모델 캐시 (빌드 타임에 다운로드하면 재시작 시 빠름)
# docker build 시 인터넷이 있으면 아래 활성화
# RUN python -c "from transformers import AutoImageProcessor, AutoModelForDepthEstimation; \
#   AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf'); \
#   AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf')"

EXPOSE 8002
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "info"]
