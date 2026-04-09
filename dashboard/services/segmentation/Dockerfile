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

EXPOSE 8003
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--log-level", "info"]
