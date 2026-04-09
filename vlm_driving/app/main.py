"""
Qwen2-VL-2B QLoRA — FastAPI 서빙 앱
실행: uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import io
import base64
import time
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ─── 설정 ─────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = Path(__file__).parent.parent / "lora_adapter"

# ─── 전역 모델 (lifespan에서 한 번만 로드) ────────────────
_model     = None
_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor
    print("[startup] 모델 로드 중...")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    _model.eval()
    _processor = AutoProcessor.from_pretrained(str(ADAPTER_PATH), trust_remote_code=True)

    print("[startup] 모델 로드 완료!")
    yield
    print("[shutdown] 종료")


app = FastAPI(
    title="Autonomous Driving VQA API",
    description="Qwen2-VL-2B QLoRA 파인튜닝 — 자율주행 장면 이해",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── 추론 함수 ────────────────────────────────────────────
@torch.inference_mode()
def _infer(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ],
    }]
    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = _processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    output_ids = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return _processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ─── 엔드포인트 ───────────────────────────────────────────
@app.get("/health")
def health():
    """서버 상태 확인"""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "device": str(next(_model.parameters()).device) if _model else "none",
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="자율주행 카메라 이미지"),
    question: str = Form(
        default="이 자율주행 차량 카메라 영상을 설명해주세요.",
        description="질문",
    ),
    max_new_tokens: int = Form(default=128, ge=16, le=512),
):
    """
    이미지 + 질문 → 자율주행 장면 분석 답변

    - **file**: JPG/PNG 이미지
    - **question**: 자연어 질문 (기본값: 장면 설명)
    - **max_new_tokens**: 최대 생성 토큰 수
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    # 이미지 로드
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")

    # 추론
    t0 = time.perf_counter()
    answer = _infer(image, question, max_new_tokens)
    elapsed = time.perf_counter() - t0

    return JSONResponse({
        "question": question,
        "answer":   answer,
        "elapsed_sec": round(elapsed, 3),
        "image_size": list(image.size),
    })


@app.post("/predict/batch")
async def predict_batch(
    file: UploadFile = File(...),
):
    """
    자율주행 7가지 표준 질문을 한 번에 실행

    이미지 한 장 → 7가지 VQA 결과 반환
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로드되지 않았습니다.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {e}")

    questions = {
        "scene_description":     "이 자율주행 차량 카메라 영상을 설명해주세요.",
        "danger_assessment":     "현재 주행 상황에서 즉각적인 위험 요소가 있습니까?",
        "action_recommendation": "현재 상황에서 자율주행 시스템이 취해야 할 행동을 알려주세요.",
        "pedestrian_check":      "현재 장면에 보행자가 있습니까?",
        "nearest_object":        "카메라에서 가장 가까운 객체는 무엇이고 거리는 얼마입니까?",
        "object_count":          "현재 장면에서 감지된 차량과 보행자의 총 수는 얼마입니까?",
        "safety_distance":       "전방 차량과의 안전 거리가 확보되어 있습니까?",
    }

    t0 = time.perf_counter()
    results = {}
    for qa_type, q in questions.items():
        results[qa_type] = _infer(image, q)
    elapsed = time.perf_counter() - t0

    return JSONResponse({
        "results":     results,
        "elapsed_sec": round(elapsed, 3),
        "image_size":  list(image.size),
    })
