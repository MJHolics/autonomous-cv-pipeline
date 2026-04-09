"""
Qwen2-VL-2B QLoRA VLM Service — 포트 8004
POST /run  : 이미지 → 7가지 자율주행 VQA 답변
GET  /health

기존 vlm_driving/app/main.py 기반으로 이식.
엔드포인트: /predict → /run (Gateway 통일 인터페이스)
"""
import io, base64, time
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ── 설정 ──────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2-VL-2B-Instruct"
ADAPTER_PATH = Path("/lora_adapter")

# 7가지 표준 VQA 질문
VQA_QUESTIONS = {
    "scene_description":     "Look at this image carefully. Describe exactly what you see: road type, vehicles, pedestrians, traffic signs, weather, and environment.",
    "danger_assessment":     "Look at this image. Are there any immediate hazards visible? Answer Yes or No, then describe what you see.",
    "action_recommendation": "Based on what is visible in this image, what should a self-driving car do right now? Be specific about what you observe.",
    "pedestrian_check":      "Look at this image. Can you see any pedestrians? Answer Yes or No, and describe their location if present.",
    "nearest_object":        "Look at this image. What object appears closest to the camera? Describe its type and approximate position in the frame.",
    "object_count":          "Count the vehicles and pedestrians visible in this image. Give a number for each.",
    "safety_distance":       "Look at this image. Is there a vehicle directly ahead? If yes, does the gap look safe or unsafe?",
}

_model     = None
_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _processor
    print(f"[vlm] 모델 로드 중: {BASE_MODEL} + {ADAPTER_PATH}")

    base = Qwen2VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model = PeftModel.from_pretrained(base, str(ADAPTER_PATH))
    _model.eval()
    _processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("[vlm] 로드 완료!")
    yield
    print("[vlm] 종료")


app = FastAPI(title="VLM Service", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── 추론 ──────────────────────────────────────────────────
@torch.inference_mode()
def _infer(image: Image.Image, question: str, max_new_tokens: int = 128) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": question},
        ],
    }]
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = _processor(
        text=[text], images=image_inputs,
        padding=True, return_tensors="pt",
    ).to(next(_model.parameters()).device)

    output_ids = _model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
    )
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return _processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "vlm",
        "model_loaded": _model is not None,
        "base_model": BASE_MODEL,
        "adapter_path": str(ADAPTER_PATH),
    }


@app.post("/run")
async def run(file: UploadFile = File(...)):
    """이미지 한 장 → 7가지 VQA 결과 반환"""
    if _model is None:
        raise HTTPException(status_code=503, detail="모델 로드 중")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    t0 = time.perf_counter()
    results = {}
    for qa_type, question in VQA_QUESTIONS.items():
        results[qa_type] = _infer(image, question)
    elapsed = round(time.perf_counter() - t0, 3)

    return JSONResponse({
        "results": results,
        "elapsed_sec": elapsed,
        "image_size": list(image.size),
    })


@app.post("/run/single")
async def run_single(
    file: UploadFile = File(...),
    question: str = Form(default="이 자율주행 카메라 영상을 설명해주세요."),
    max_new_tokens: int = Form(default=128),
):
    """단일 질문 모드"""
    if _model is None:
        raise HTTPException(status_code=503, detail="모델 로드 중")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    t0 = time.perf_counter()
    answer = _infer(image, question, max_new_tokens)
    elapsed = round(time.perf_counter() - t0, 3)

    return JSONResponse({
        "question": question,
        "answer": answer,
        "elapsed_sec": elapsed,
    })
