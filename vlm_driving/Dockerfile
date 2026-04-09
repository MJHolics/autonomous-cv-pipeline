# Qwen2-VL-2B QLoRA 서빙 — Docker 이미지
# 빌드: docker build -t vlm-driving .
# 실행: docker run --gpus all -p 8000:8000 vlm-driving

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드 + LoRA 어댑터 복사
COPY app/ ./app/
COPY lora_adapter/ ./lora_adapter/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
