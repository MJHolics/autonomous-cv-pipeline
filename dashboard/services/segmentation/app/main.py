"""
SegFormer Segmentation Service — 포트 8003
POST /run  : 이미지 → semantic 색상 오버레이(base64) + 클래스 분포
GET  /health

모델: nvidia/segformer-b0-finetuned-ade-512-512 (ADE20K pretrained)
참고: CARLA 파인튜닝 체크포인트가 디스크에 없어 pretrained 사용.
      DEVLOG.md 결정 1 참고.
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
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ── 설정 ──────────────────────────────────────────────────
MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ADE20K 주요 클래스 색상 (150개 중 주행 관련 강조)
# fmt: off
ADE20K_COLORS = np.array([
    [120, 120, 120], [180, 120, 120], [6, 230, 230],   [80, 50, 50],
    [4, 200, 3],     [120, 120, 80],  [140, 140, 140], [204, 5, 255],
    [230, 230, 230], [4, 250, 7],     [224, 5, 255],   [235, 255, 7],
    [150, 5, 61],    [120, 120, 70],  [8, 255, 51],    [255, 6, 82],
    [143, 255, 140], [204, 255, 4],   [255, 51, 7],    [204, 70, 3],
    [0, 102, 200],   [61, 230, 250],  [255, 6, 51],    [11, 102, 255],
    [255, 7, 71],    [255, 9, 224],   [9, 7, 230],     [220, 220, 220],
    [255, 9, 92],    [112, 9, 255],   [8, 255, 214],   [7, 255, 224],
    [255, 184, 6],   [10, 255, 71],   [255, 41, 10],   [7, 255, 255],
    [224, 255, 8],   [102, 8, 255],   [255, 61, 6],    [255, 194, 7],
    [255, 122, 8],   [0, 255, 20],    [255, 8, 41],    [255, 5, 153],
    [6, 51, 255],    [235, 12, 255],  [160, 150, 20],  [0, 163, 255],
    [140, 140, 140], [250, 10, 15],   [20, 255, 0],    [31, 255, 0],
    [255, 31, 0],    [255, 224, 0],   [153, 255, 0],   [0, 0, 255],
    [255, 71, 0],    [0, 235, 255],   [0, 173, 255],   [31, 0, 255],
    [11, 200, 200],  [255, 82, 0],    [0, 255, 245],   [0, 61, 255],
    [0, 255, 112],   [0, 255, 133],   [255, 0, 0],     [255, 163, 0],
    [255, 102, 0],   [194, 255, 0],   [0, 143, 255],   [51, 255, 0],
    [0, 82, 255],    [0, 255, 41],    [0, 255, 173],   [10, 0, 255],
    [173, 255, 0],   [0, 255, 153],   [255, 92, 0],    [255, 0, 255],
    [255, 0, 245],   [255, 0, 102],   [255, 173, 0],   [255, 0, 20],
    [255, 184, 184], [0, 31, 255],    [0, 255, 61],    [0, 71, 255],
    [255, 0, 204],   [0, 255, 194],   [0, 255, 82],    [0, 10, 255],
    [0, 112, 255],   [51, 0, 255],    [0, 194, 255],   [0, 122, 255],
    [0, 255, 163],   [255, 153, 0],   [0, 255, 10],    [255, 112, 0],
    [143, 255, 0],   [82, 0, 255],    [163, 255, 0],   [255, 235, 0],
    [8, 184, 170],   [133, 0, 255],   [0, 255, 92],    [184, 0, 255],
    [255, 0, 31],    [0, 184, 255],   [0, 214, 255],   [255, 0, 112],
    [92, 255, 0],    [0, 224, 255],   [112, 224, 255], [70, 184, 160],
    [163, 0, 255],   [153, 0, 255],   [71, 255, 0],    [255, 0, 163],
    [255, 204, 0],   [255, 0, 143],   [0, 255, 235],   [133, 255, 0],
    [255, 0, 235],   [245, 0, 255],   [255, 0, 122],   [255, 245, 0],
    [10, 190, 212],  [214, 255, 0],   [0, 204, 255],   [20, 0, 255],
    [255, 255, 0],   [0, 153, 255],   [0, 41, 255],    [0, 255, 204],
    [41, 0, 255],    [41, 255, 0],    [173, 0, 255],   [0, 245, 255],
    [71, 0, 255],    [122, 0, 255],   [0, 255, 184],   [0, 92, 255],
    [184, 255, 0],   [0, 133, 255],   [255, 214, 0],   [25, 194, 194],
    [102, 255, 0],   [92, 0, 255],
], dtype=np.uint8)
# fmt: on

_processor = None
_model = None
_id2label = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processor, _model, _id2label
    print(f"[segmentation] 모델 로드 중: {MODEL_ID} on {DEVICE}")
    _processor = SegformerImageProcessor.from_pretrained(MODEL_ID)
    _model = SegformerForSemanticSegmentation.from_pretrained(MODEL_ID)
    _model = _model.to(DEVICE).eval()
    _id2label = _model.config.id2label
    print(f"[segmentation] 로드 완료 — {len(_id2label)}개 클래스")
    yield
    print("[segmentation] 종료")


app = FastAPI(title="Segmentation Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── 추론 ──────────────────────────────────────────────────
@torch.inference_mode()
def _predict(image: Image.Image) -> np.ndarray:
    inputs = _processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = _model(**inputs)
    upsampled = F.interpolate(
        outputs.logits,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.int16)


def _seg_to_base64(seg: np.ndarray, original: Image.Image) -> tuple[str, dict]:
    """seg map → 색상 오버레이 이미지(base64) + 클래스 분포"""
    h, w = seg.shape
    rgb = ADE20K_COLORS[seg % len(ADE20K_COLORS)]

    # 원본과 50% 블렌딩
    orig_arr = np.array(original.resize((w, h)))
    blended = (orig_arr * 0.45 + rgb * 0.55).astype(np.uint8)

    # 클래스 분포 (상위 5개)
    unique, counts = np.unique(seg, return_counts=True)
    total = seg.size
    top5 = sorted(
        [{"id": int(c), "label": _id2label.get(int(c), f"class_{c}"),
          "pct": round(float(cnt / total * 100), 1)}
         for c, cnt in zip(unique, counts)],
        key=lambda x: x["pct"], reverse=True
    )[:5]

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    axes[0].imshow(original)
    axes[0].set_title("원본", color="white", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(blended)
    axes[1].set_title("Semantic Segmentation", color="white", fontsize=12)
    axes[1].axis("off")
    plt.tight_layout(pad=1.0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return base64.b64encode(buf.getvalue()).decode(), top5


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "segmentation",
        "model_loaded": _model is not None,
        "device": DEVICE,
        "num_classes": len(_id2label),
    }


@app.post("/run")
async def run(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="모델 로드 중")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    t0 = time.perf_counter()
    seg = _predict(image)
    elapsed = round(time.perf_counter() - t0, 3)

    seg_b64, top_classes = _seg_to_base64(seg, image)

    return JSONResponse({
        "seg_image": seg_b64,
        "top_classes": top_classes,
        "elapsed_sec": elapsed,
        "image_size": list(image.size),
    })
