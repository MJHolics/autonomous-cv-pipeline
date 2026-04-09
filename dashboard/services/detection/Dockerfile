FROM python:3.11-slim

WORKDIR /service

# 시스템 의존성 (OpenCV, libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    # GPU 지원 torch (CUDA 12.4)
    && pip install --no-cache-dir \
       torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
       --extra-index-url https://download.pytorch.org/whl/cu124

COPY app/ ./app/

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--log-level", "info"]
