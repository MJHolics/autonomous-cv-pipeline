"""
YOLOv8 Detection Service — 포트 8001
POST /run  : 이미지 → bbox 리스트 + 오버레이 이미지(base64)
GET  /health
"""
import io, base64, time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────
MODEL_PATH = Path("/models/best.pt")

# CARLA finetuned 클래스 색상 (클래스 ID 순서대로)
CLASS_COLORS = [
    (255, 56, 56),   (255, 157, 151), (255, 112, 31),  (255, 178, 29),
    (207, 210, 49),  (72, 249, 10),   (146, 204, 23),  (61, 219, 134),
    (26, 147, 52),   (0, 212, 187),   (44, 153, 168),  (0, 194, 255),
    (52, 69, 147),   (100, 115, 255), (0, 24, 236),    (132, 56, 255),
    (82, 0, 133),    (203, 56, 255),  (255, 149, 200), (255, 55, 199),
]

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    print(f"[detection] 모델 로드 중: {MODEL_PATH}")
    _model = YOLO(str(MODEL_PATH))
    print(f"[detection] 로드 완료 — 클래스: {list(_model.names.values())[:5]}...")
    yield
    print("[detection] 종료")


app = FastAPI(title="Detection Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── 시각화 ────────────────────────────────────────────────
def _draw_boxes(image: Image.Image, boxes: list) -> str:
    """bbox 오버레이 → base64 PNG"""
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for box in boxes:
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
        cls = box["cls"]
        label = f"{box['label']} {box['conf']:.2f}"
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]

        # bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 라벨 배경
        text_bbox = draw.textbbox((x1, y1 - 18), label)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 18), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "detection",
        "model_loaded": _model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/run")
async def run(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="모델 로드 중")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_arr = np.array(image)

    t0 = time.perf_counter()
    results = _model.predict(source=img_arr, conf=0.25, verbose=False)
    elapsed = round(time.perf_counter() - t0, 3)

    # bbox 추출
    boxes = []
    result = results[0]
    names = _model.names
    for box in result.boxes:
        x1, y1, x2, y2 = [round(v, 1) for v in box.xyxy[0].cpu().tolist()]
        cls_id = int(box.cls[0])
        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": round(float(box.conf[0]), 3),
            "cls": cls_id,
            "label": names.get(cls_id, f"class_{cls_id}"),
        })

    overlay_b64 = _draw_boxes(image, boxes)

    return JSONResponse({
        "boxes": boxes,
        "overlay_image": overlay_b64,
        "count": len(boxes),
        "elapsed_sec": elapsed,
        "image_size": list(image.size),
    })
