FROM python:3.11-slim

WORKDIR /service

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
       torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
       --extra-index-url https://download.pytorch.org/whl/cu124

COPY app/ ./app/

# LoRA 어댑터는 docker-compose에서 볼륨으로 마운트
# /lora_adapter → ../../vlm_driving/lora_adapter

EXPOSE 8004
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--log-level", "info"]
