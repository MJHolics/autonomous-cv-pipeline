"""
API Gateway — 포트 8000
POST /analyze       : 이미지 → 4개 서비스 병렬 호출 → 통합 결과
GET  /health        : 전체 서비스 상태 확인
GET  /samples       : CARLA 샘플 이미지 목록
GET  /samples/{name}: 샘플 이미지 파일 반환
"""
import asyncio, base64, io, os, time
from pathlib import Path

import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# ── 서비스 URL (환경변수 → 기본값) ───────────────────────
DETECTION_URL    = os.getenv("DETECTION_URL",    "http://detection:8001")
DEPTH_URL        = os.getenv("DEPTH_URL",        "http://depth:8002")
SEGMENTATION_URL = os.getenv("SEGMENTATION_URL", "http://segmentation:8003")
VLM_URL          = os.getenv("VLM_URL",          "http://vlm:8004")

SAMPLE_DIR = Path(os.getenv("SAMPLE_DIR", "/carla_samples"))

SERVICES = {
    "detection":    DETECTION_URL,
    "depth":        DEPTH_URL,
    "segmentation": SEGMENTATION_URL,
    "vlm":          VLM_URL,
}

# ── HTTP 클라이언트 ────────────────────────────────────────
_client: httpx.AsyncClient | None = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=120.0)
    return _client


app = FastAPI(title="Autonomous Driving Dashboard — Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 헬퍼: 단일 서비스 호출 ────────────────────────────────
async def _call_service(
    client: httpx.AsyncClient,
    url: str,
    img_bytes: bytes,
    filename: str,
) -> dict:
    try:
        resp = await client.post(
            f"{url}/run",
            files={"file": (filename, img_bytes, "image/jpeg")},
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.TimeoutException:
        return {"error": "timeout"}
    except Exception as e:
        return {"error": str(e)}


# ── 엔드포인트 ────────────────────────────────────────────
@app.get("/health")
async def health():
    """모든 서비스의 health 상태를 한번에 확인"""
    client = await get_client()
    results = {}

    async def _check(name: str, url: str):
        try:
            r = await client.get(f"{url}/health", timeout=5.0)
            results[name] = r.json() if r.status_code == 200 else {"status": "error", "code": r.status_code}
        except Exception as e:
            results[name] = {"status": "unreachable", "error": str(e)}

    await asyncio.gather(*[_check(n, u) for n, u in SERVICES.items()])

    all_ok = all(v.get("status") == "ok" for v in results.values())
    return JSONResponse({
        "gateway": "ok",
        "all_services_ok": all_ok,
        "services": results,
    })


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    이미지 업로드 → 4개 서비스 병렬 호출 → 통합 결과 반환

    응답 구조:
    {
      "detection":    { "boxes": [...], "overlay_image": "base64...", "elapsed_sec": 0.3 },
      "depth":        { "depth_image": "base64...", "stats": {...},   "elapsed_sec": 0.5 },
      "segmentation": { "seg_image":   "base64...", "top_classes": [...], "elapsed_sec": 0.4 },
      "vlm":          { "results": {...},            "elapsed_sec": 4.2 },
      "total_elapsed_sec": 4.5
    }
    """
    img_bytes = await file.read()
    filename  = file.filename or "image.jpg"
    client    = await get_client()

    # 4개 서비스 동시 호출
    t0 = time.perf_counter()
    det_task, dep_task, seg_task, vlm_task = await asyncio.gather(
        _call_service(client, DETECTION_URL,    img_bytes, filename),
        _call_service(client, DEPTH_URL,        img_bytes, filename),
        _call_service(client, SEGMENTATION_URL, img_bytes, filename),
        _call_service(client, VLM_URL,          img_bytes, filename),
    )
    total = round(time.perf_counter() - t0, 3)

    return JSONResponse({
        "detection":    det_task,
        "depth":        dep_task,
        "segmentation": seg_task,
        "vlm":          vlm_task,
        "total_elapsed_sec": total,
    })


@app.get("/samples")
async def list_samples():
    """CARLA 샘플 이미지 목록 (파일명 리스트)"""
    if not SAMPLE_DIR.exists():
        return JSONResponse({"samples": [], "error": "샘플 디렉토리 없음"})

    samples = sorted([
        f.name for f in SAMPLE_DIR.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])[:30]  # 최대 30개
    return JSONResponse({"samples": samples, "count": len(samples)})


@app.get("/samples/{name}")
async def get_sample(name: str):
    """특정 샘플 이미지 파일 반환"""
    # 경로 순회 공격 방지
    safe_name = Path(name).name
    path = SAMPLE_DIR / safe_name
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="샘플 이미지 없음")
    return FileResponse(str(path), media_type="image/jpeg")


@app.get("/samples/{name}/base64")
async def get_sample_b64(name: str):
    """샘플 이미지를 base64로 반환 (프론트엔드 미리보기용)"""
    safe_name = Path(name).name
    path = SAMPLE_DIR / safe_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="샘플 이미지 없음")
    data = base64.b64encode(path.read_bytes()).decode()
    return JSONResponse({"name": safe_name, "data": f"data:image/jpeg;base64,{data}"})
