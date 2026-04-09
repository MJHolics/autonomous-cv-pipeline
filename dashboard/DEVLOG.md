# Dashboard 개발 로그 (Claude 자율 작업)

> 사장님 자리 비우신 동안 작업한 내용을 이유와 함께 기록합니다.  
> 시간순 기록 — 중요한 결정에는 ⚠️ 표시.

---

## 2026-04-02 — 초기 구현

### ⚠️ 결정 1: SegFormer 체크포인트 없음 → ADE20K pretrained 사용

**상황**: `phase4_carla/finetuning/`에 SegFormer 체크포인트(`checkpoints/segformer_carla/`)가 존재하지 않음.  
노트북(`03_segformer_carla_semantic.ipynb`)에서 학습은 완료했지만 `save_pretrained()` 호출이 없거나 경로가 달랐던 것으로 추정.

**결정**: `nvidia/segformer-b0-finetuned-ade-512-512` (ADE20K 150 클래스 pretrained) 사용.  
**이유**: ADE20K는 road, building, car, person, sky, vegetation 등 자율주행 관련 클래스 포함. 데모 시각화 목적으로는 충분.  
**한계**: CARLA 파인튜닝 수치(mIoU 0.586)는 반영 안 됨. VLM/Detection 결과가 메인 스토리이므로 허용.  
**향후**: 노트북에서 `model.save_pretrained("checkpoints/segformer_carla")` 재실행 후 경로 교체 가능.

---

### ⚠️ 결정 2: Depth 모델 선택 → Metric-Outdoor 사용

**상황**: DepthAnythingV2에 두 가지 옵션 존재.
- `Depth-Anything-V2-Small-hf`: 상대 disparity (단위 없음, 정렬 필요)
- `Depth-Anything-V2-Metric-Outdoor-Small-hf`: 절대 미터 단위

**결정**: Metric-Outdoor 사용.  
**이유**: 
1. CARLA 환경(야외 주행) = Outdoor 도메인과 일치
2. 정렬 코드 불필요 → 서비스 단순화
3. 응답에 실제 미터 수치를 포함할 수 있어 포트폴리오 데모에 더 인상적
4. Phase 4 스케일 정렬 결과(RMSE 2.71m)와 비슷한 수준 기대

---

### ⚠️ 결정 3: Detection 모델 → CARLA 파인튜닝 가중치 사용

**경로**: `phase4_carla/finetuning/runs/carla_finetune/weights/best.pt`  
**이유**: mAP 0.68 (파인튜닝 후)로 CARLA 도메인에서 의미 있는 성능. 포트폴리오 스토리("도메인 적응 전후 비교")와 직결.

---

### ⚠️ 결정 4: VLM 서비스 포트 변경

`vlm_driving/app/main.py` 기존 포트: 8000  
Dashboard 구조: gateway가 8000 사용  
**변경**: VLM 서비스 → 포트 8004

---

### ⚠️ 결정 5: Docker GPU 전략

RTX 4080 SUPER 16GB에서 4개 모델 동시 로드:
- YOLOv8s CARLA: ~0.5GB
- DepthAnythingV2-Small: ~0.5GB  
- SegFormer-B0: ~0.3GB
- Qwen2-VL-2B (4bit): ~4.5GB
- 합계: ~5.8GB → 16GB 내 안전

**전략**: 모든 컨테이너가 같은 GPU 공유 (Docker `device_ids: ["0"]`)  
VLM은 `deploy.resources.limits` 없이 필요한 만큼 사용하되 4-bit 양자화로 제한.

---

### ⚠️ 결정 6: 프론트엔드 스타일링 → 순수 CSS (Tailwind 없음)

**이유**: Tailwind CSS는 PostCSS 설정과 npm 패키지 추가 필요. 바이브 코딩 특성상 CSS를 직접 작성하는 게 더 안정적이고 빠름.  
**스타일**: 다크 모드, 자동차/테크 느낌 (`#0d1117` 배경, `#00ff88` 액센트).

---

### ⚠️ 결정 7: 응답 포맷에 base64 PNG 포함

각 서비스가 추론 결과 외에 **시각화 이미지를 base64로 반환**.  
**이유**: 프론트엔드에서 별도 이미지 처리 없이 `<img src="data:image/png;base64,...">` 직접 사용.  
**트레이드오프**: 응답 크기 증가 (이미지 1장 ~100-300KB) → 로컬 데모에서는 허용.

---

## 구현 파일 목록

```
dashboard/
├── DEVLOG.md              ← 이 파일
├── plan.md                ← 기획서 (체크박스 업데이트)
├── services/
│   ├── detection/         ← YOLOv8 CARLA finetuned (포트 8001)
│   ├── depth/             ← DepthAnythingV2 Metric-Outdoor (포트 8002)
│   ├── segmentation/      ← SegFormer ADE20K pretrained (포트 8003)
│   ├── vlm/               ← Qwen2-VL-2B QLoRA (포트 8004)
│   └── gateway/           ← API Gateway, asyncio.gather() (포트 8000)
├── frontend/              ← React + Vite + TypeScript
└── docker-compose.yml
```

## 구현 완료 현황 (2026-04-02)

| 파일 | 상태 |
|------|------|
| `services/detection/app/main.py` | ✅ YOLOv8 CARLA best.pt, bbox 오버레이 |
| `services/depth/app/main.py` | ✅ DepthAnythingV2 Metric, plasma 히트맵 |
| `services/segmentation/app/main.py` | ✅ SegFormer ADE20K, 50% 블렌딩 오버레이 |
| `services/vlm/app/main.py` | ✅ Qwen2-VL-2B QLoRA, 7가지 VQA |
| `services/gateway/app/main.py` | ✅ asyncio.gather() 병렬 호출, 샘플 API |
| `services/*/Dockerfile` | ✅ python:3.11-slim + CUDA 12.4 |
| `services/*/requirements.txt` | ✅ 각 서비스별 최소 의존성 |
| `frontend/src/App.tsx` | ✅ 2컬럼 레이아웃, 서비스 상태 인디케이터 |
| `frontend/src/components/ResultTabs.tsx` | ✅ 5개 탭, 위험 배너, 통계 카드 |
| `frontend/src/components/ImageUploader.tsx` | ✅ 드래그앤드롭 |
| `frontend/src/components/SamplePicker.tsx` | ✅ CARLA 샘플 30개 버튼 |
| `frontend/src/api/client.ts` | ✅ 타입 정의 + 모든 API 함수 |
| `frontend/Dockerfile` | ✅ 멀티스테이지 빌드 (node → nginx) |
| `frontend/nginx.conf` | ✅ SPA 라우팅 + /api 프록시 |
| `docker-compose.yml` | ✅ 6개 서비스, GPU, HF 캐시 공유 볼륨 |

## 사장님이 돌아오시면 할 일

### 필수 (순서대로)

1. **프론트엔드 의존성 설치**
   ```bash
   cd dashboard/frontend
   npm install
   ```

2. **Docker Compose로 전체 실행**
   ```bash
   cd dashboard
   docker compose up --build
   ```
   - 첫 빌드: 20~40분 (모델 다운로드 + pip install 포함)
   - 이후 재실행: 2~3분

3. **브라우저에서 확인**
   - 대시보드: `http://localhost:3000`
   - API 문서: `http://localhost:8000/docs`
   - 서비스 상태: `http://localhost:8000/health`

4. **CARLA 샘플 클릭** → Detection + Depth + Segmentation + VLM 동시 분석 확인

### 선택 (향상 작업)

5. **SegFormer CARLA 파인튜닝 체크포인트 복구**
   - `phase4_carla/finetuning/03_segformer_carla_semantic.ipynb` 마지막 셀에 추가:
     ```python
     model.save_pretrained("checkpoints/segformer_carla")
     ```
   - `services/segmentation/app/main.py` 의 `MODEL_ID` 변경:
     ```python
     MODEL_ID = "/checkpoints/segformer_carla"  # 로컬 경로
     ```
   - `docker-compose.yml` segmentation 서비스에 볼륨 추가:
     ```yaml
     - ../phase4_carla/finetuning/checkpoints/segformer_carla:/checkpoints/segformer_carla:ro
     ```

6. **동영상 처리 기능 추가 (v2)**
   - Gateway에 `POST /analyze/video` 엔드포인트 추가
   - 프레임 추출 → 배치 처리 → 결과 동영상 생성
