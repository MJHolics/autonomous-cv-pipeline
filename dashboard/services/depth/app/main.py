"""
DepthAnythingV2 Depth Service — 포트 8002
POST /run  : 이미지 → depth 히트맵(base64) + 통계
GET  /health

모델: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf
이유: Metric 모델은 절대 미터 단위 출력 → 정렬 코드 불필요
"""
import io, base64, time
from contextlib import asynccontextmanager

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ── 설정 ──────────────────────────────────────────────────
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_processor = None
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model
    print(f"[depth] 모델 로드 중: {MODEL_ID} on {DEVICE}")
    _processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    _model = _model.to(DEVICE).eval()
    print("[depth] 로드 완료!")
    yield
    print("[depth] 종료")


app = FastAPI(title="Depth Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── 추론 ──────────────────────────────────────────────────
@torch.inference_mode()
def _predict(image: Image.Image) -> np.ndarray:
    """Metric depth map (미터 단위) 반환"""
    inputs = _processor(images=image, return_tensors="pt").to(DEVICE)
    outputs = _model(**inputs)
    depth = F.interpolate(
        outputs.predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()
    return depth.astype(np.float32)


def _depth_to_base64(depth: np.ndarray, original: Image.Image) -> str:
    """Depth map → plasma 컬러맵 + 원본 나란히 → base64 PNG"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    # 원본
    axes[0].imshow(original)
    axes[0].set_title("원본", color="white", fontsize=12)
    axes[0].axis("off")

    # Depth 히트맵
    valid = depth[depth > 0]
    vmin = float(np.percentile(valid, 5)) if len(valid) > 0 else 0
    vmax = float(np.percentile(valid, 95)) if len(valid) > 0 else 50

    im = axes[1].imshow(depth, cmap="plasma_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Depth Map (m)", color="white", fontsize=12)
    axes[1].axis("off")

    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("거리 (m)", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout(pad=1.0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "depth",
        "model_loaded": _model is not None,
        "device": DEVICE,
        "model_id": MODEL_ID,
    }


@app.post("/run")
async def run(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="모델 로드 중")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    t0 = time.perf_counter()
    depth = _predict(image)
    elapsed = round(time.perf_counter() - t0, 3)

    depth_b64 = _depth_to_base64(depth, image)

    valid = depth[depth > 0]
    return JSONResponse({
        "depth_image": depth_b64,
        "stats": {
            "min_m": round(float(valid.min()), 2) if len(valid) else 0,
            "max_m": round(float(valid.max()), 2) if len(valid) else 0,
            "mean_m": round(float(valid.mean()), 2) if len(valid) else 0,
            "median_m": round(float(np.median(valid)), 2) if len(valid) else 0,
        },
        "elapsed_sec": elapsed,
        "image_size": list(image.size),
    })
