# 실시간 주행 장면 분석 대시보드 — 구현 계획

> **작성일**: 2026-04-02  
> **목표**: Phase 1~4 + VLM 파이프라인을 하나의 웹 대시보드로 통합  
> **방식**: 인라인 메모를 직접 달아서 검토 후 구현 시작  
>
> ✏️ **메모 사용법**: 이 파일을 에디터에서 열고 각 항목 옆에 직접 메모를 달면 됩니다.  
> 예) `<!-- 이건 제외 -->`, `<!-- 이 부분 먼저 -->`, `<!-- 라이브러리 바꿔줘 -->`  
> 메모 다 달고 나서 "메모 반영해서 계획 업데이트해줘"라고 하면 됩니다.

---

## 0. 완성 기준 (Definition of Done)

- [x] 이미지 or 동영상 업로드 → Detection + Depth + Segmentation + VLM 결과가 한 화면에 표시
- [x] 각 모델 추론 결과가 시각적으로 오버레이된 이미지 제공
- [x] FastAPI 백엔드 + 프론트엔드 UI
- [x] Docker Compose로 단일 명령 실행 (`docker compose up`)
- [x] 로컬 CARLA 데이터셋으로 즉시 테스트 가능

---

## 1. 아키텍처 선택지

### 1-A. 프론트엔드 선택 <!-- ✏️ 여기에 선택 메모 달아주세요 --> React + Vite

| 옵션 | 장점 | 단점 |
|------|------|------|
| **Streamlit** | 파이썬만으로 완성, 설치 간단 | 레이아웃 자유도 낮음, 느림 |
| **Gradio** | 파이썬만으로 완성, 이미지 UI 특화 | 커스터마이징 제한 |
| **React + Vite** | 자유로운 레이아웃, 실시간 WebSocket | JS 코드 필요 |
| **Next.js** | SSR, 파일 구조 정돈 | 러닝 커브 |

> ✅ **확정: React + Vite**  
> 자유로운 레이아웃 + WebSocket 확장성. 바이브 코딩으로 UI는 Claude가 생성하므로 JS 코드 부담 없음.

### 1-B. 백엔드 구조 <!-- ✏️ 단일 FastAPI vs 마이크로서비스 선택 메모 달아주세요 --> 마이크로서비스

| 옵션 | 설명 |
|------|------|
| **단일 FastAPI** | 모든 모델을 하나의 프로세스에서 실행 (현재 vlm_driving/app/main.py 확장) |
| **마이크로서비스** | Detection/Depth/Seg/VLM 각각 별도 포트 (Docker Compose로 관리) |

> ✅ **확정: 마이크로서비스**  
> Detection / Depth / Seg / VLM 각각 독립 컨테이너. Docker Compose로 관리.  
> 장점: VRAM 부족 시 특정 서비스만 재시작, 각 모델 버전 독립 관리 가능.  
> 주의: 서비스 간 통신은 HTTP (내부 Docker 네트워크), Gateway 역할의 API 서비스 별도 추가 필요.

### 1-C. 실시간 처리 방식 <!-- ✏️ 이미지 업로드만 vs 동영상/웹캠도 포함 여부 메모 달아주세요 --> 사진 + 동영상은 선택 확장

| 옵션 | 설명 |
|------|------|
| **이미지 업로드** | 정적 이미지 한 장 업로드 → 분석 결과 반환 |
| **동영상 업로드** | mp4 업로드 → 프레임별 처리 → 결과 동영상 다운로드 |
| **실시간 웹캠** | WebSocket 스트리밍 (구현 복잡도 높음) |

> ✅ **확정: 이미지 업로드 (MVP)**, 동영상은 v2 선택 확장  

---

## 2. 구현 범위 (MVP)

> MVP = 가장 빠르게 "돌아가는 데모"를 만드는 최소 범위  
> 아래 체크박스는 구현 우선순위 기준 — 체크 없는 항목은 v2에서 구현

### 백엔드 엔드포인트 <!-- ✏️ 불필요한 엔드포인트는 "제외" 메모 달아주세요 -->

- [ ] `POST /analyze` — 이미지 업로드 → 4개 모델 전부 실행 → 통합 결과 반환
- [ ] `POST /analyze/detection` — Detection만 단독 실행
- [ ] `POST /analyze/depth` — Depth만 단독 실행  
- [ ] `POST /analyze/segmentation` — Segmentation만 단독 실행
- [ ] `POST /analyze/vlm` — VLM만 단독 실행 (기존 `/predict` 대체)
- [ ] `GET /health` — 각 모델 로드 상태 확인
- [ ] `GET /models` — 현재 로드된 모델 목록 + VRAM 사용량

### 프론트엔드 화면 <!-- ✏️ 레이아웃 관련 메모 달아주세요 -->

- [ ] 이미지 업로드 영역 (드래그앤드롭 or 파일 선택)
- [ ] CARLA 샘플 이미지 선택 버튼 (빠른 테스트용)
- [ ] 분석 결과 탭 구성:
  - [ ] **Overview 탭**: 4개 모델 결과 한눈에 (2x2 그리드)
  - [ ] **Detection 탭**: YOLOv8 bbox 오버레이 + confidence 수치
  - [ ] **Depth 탭**: DepthAnythingV2 히트맵 컬러맵
  - [ ] **Segmentation 탭**: Semantic 색상 오버레이
  - [ ] **VLM 탭**: 7가지 질문 답변 텍스트
- [ ] 추론 시간 표시 (모델별)

---

## 3. 기존 코드 재사용 계획

> 이미 만들어진 것들 — 중복 구현 금지 <!-- ✏️ 재사용하기 어려운 게 있으면 메모 달아주세요 -->

| 기존 파일 | 재사용 방식 |
|-----------|------------|
| `vlm_driving/app/main.py` | 베이스로 사용, `/predict` → `/analyze/vlm`으로 통합 |
| `vlm_driving/lora_adapter/` | VLM 모델 가중치 그대로 사용 |
| `phase1_basics/detection/` 노트북 | YOLOv8 추론 코드 추출 |
| `phase1_basics/depth/` 노트북 | DepthAnythingV2 추론 + 시각화 코드 추출 |
| `phase4_carla/data_collection/carla_dataset/` | 샘플 이미지 소스 (테스트용) |
| `vlm_driving/Dockerfile` | 베이스로 사용, 모델 추가 |

---

## 4. 폴더 구조 (생성 예정)

> 마이크로서비스 구조: 각 모델이 독립 컨테이너로 실행

```
dashboard/
├── plan.md                        ← 이 파일
│
├── services/
│   ├── gateway/                   ← API Gateway (포트 8000)
│   │   ├── app/
│   │   │   └── main.py            ← 라우팅: /analyze → 각 서비스 호출 후 결과 합산
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── detection/                 ← YOLOv8 서비스 (포트 8001)
│   │   ├── app/
│   │   │   ├── main.py            ← POST /run → bbox + confidence 반환
│   │   │   └── visualizer.py      ← bbox 오버레이 이미지 생성
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── depth/                     ← DepthAnythingV2 서비스 (포트 8002)
│   │   ├── app/
│   │   │   ├── main.py            ← POST /run → depth map (base64 PNG) 반환
│   │   │   └── visualizer.py      ← 히트맵 컬러맵 생성
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── vlm/                       ← Qwen2-VL 서비스 (포트 8003)
│   │   ├── app/
│   │   │   └── main.py            ← POST /run → 7가지 VQA 답변 반환 (기존 코드 이식)
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── segmentation/              ← SegFormer 서비스 (포트 8004) <!-- ✏️ 포함/제외 결정 필요 -->
│       ├── app/
│       │   ├── main.py
│       │   └── visualizer.py
│       ├── Dockerfile
│       └── requirements.txt
│
├── frontend/                      ← React + Vite 앱
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── ImageUploader.tsx   ← 드래그앤드롭 업로드
│   │   │   ├── SamplePicker.tsx    ← CARLA 샘플 이미지 선택
│   │   │   ├── ResultTabs.tsx      ← Detection / Depth / VLM 탭
│   │   │   └── OverlayImage.tsx    ← 오버레이 이미지 표시
│   │   └── api/
│   │       └── client.ts           ← Gateway API 호출
│   ├── Dockerfile
│   └── package.json
│
└── docker-compose.yml             ← 전체 서비스 한 번에 실행
```

---

## 5. 구현 단계 <!-- ✏️ 순서 바꾸거나 단계 합치고 싶으면 메모 달아주세요 -->

> 각 단계가 독립적으로 테스트 가능하도록 설계. 완료 시 이 파일에 체크.

### 단계 1 — 개별 서비스 구축 (모델별 FastAPI) ✅
- [x] `services/detection/` — YOLOv8 CARLA finetuned (best.pt)
- [x] `services/depth/` — DepthAnythingV2 Metric-Outdoor
- [x] `services/vlm/` — Qwen2-VL-2B QLoRA (기존 코드 이식, /run으로 통일)
- [x] `services/segmentation/` — SegFormer ADE20K pretrained (DEVLOG 결정 1 참고)

### 단계 2 — API Gateway ✅
- [x] `services/gateway/app/main.py`
  - `POST /analyze`: asyncio.gather()로 4개 서비스 병렬 호출
  - `GET /health`: 전체 서비스 상태 확인
  - `GET /samples`, `GET /samples/{name}`, `GET /samples/{name}/base64`

### 단계 3 — React + Vite 프론트엔드 ✅
- [x] `ImageUploader.tsx` — 드래그앤드롭 업로드
- [x] `SamplePicker.tsx` — CARLA 샘플 이미지 선택 (최대 30개)
- [x] `ResultTabs.tsx` — 개요/Detection/Depth/Segmentation/VLM 5개 탭
- [x] `OverlayImage.tsx` — base64 이미지 표시
- [x] `api/client.ts` — Gateway API 타입 정의 + 호출
- [x] 추론 시간 badge (색상: green<1s, orange<5s, red>=5s)
- [x] 서비스 상태 인디케이터 (헤더 + 배너)
- [x] 포트폴리오 성과 수치 카드 (왼쪽 패널 하단)

### 단계 4 — Docker Compose 통합 ✅
- [x] 각 서비스 `Dockerfile` (python:3.11-slim + CUDA 12.4)
- [x] `frontend/Dockerfile` (node:20 빌드 → nginx:alpine 서빙)
- [x] `nginx.conf` (SPA 라우팅 + /api → gateway 프록시)
- [x] `docker-compose.yml` (5개 서비스 + GPU 마운트 + 볼륨)
- [ ] `docker compose up` 실행 확인 ← 사장님 복귀 후 실행

---

## 6. 리스크 & 결정 필요 항목 <!-- ✏️ 중요한 결정 사항 — 반드시 읽어주세요 -->

| 리스크 | 내용 | 결정 필요 |
|--------|------|-----------|
| **VRAM 부족** | YOLOv8 + DepthAnythingV2 + Qwen2-VL 동시 로드 시 16GB 초과 가능 | 마이크로서비스로 해결: GPU 컨테이너별 메모리 격리. VLM은 단독 GPU 할당 |
| **Seg 모델 포함 여부** | SegFormer는 추가 ~2GB VRAM. MVP에 반드시 필요한가? | **결정 필요** <!-- ✏️ 포함 / 제외 --> |
| **응답 시간** | Gateway가 3개 서비스를 asyncio 병렬 호출 → 가장 느린 VLM(~4초)에 수렴 | Gateway에서 `asyncio.gather()` 병렬 호출로 구현 예정 |
| **CORS** | React(3000) → Gateway(8000) 크로스 오리진 | FastAPI `CORSMiddleware` 추가 필요 (단계 2에서 처리) |
| **샘플 이미지 마운트** | CARLA 데이터셋을 컨테이너에서 접근하는 방법 | docker-compose volume 마운트: `../phase4_carla/data_collection/carla_dataset` |

---

## 7. 완료 후 포트폴리오 활용법

- GitHub README에 GIF 데모 추가 (OBS로 화면 녹화)
- `docker compose up` 한 줄로 실행되는 것 강조
- 성과 수치 (ROUGE-L 0.759, mAP 0.68 등) 대시보드 안에서 직접 확인 가능하다는 점 강조
- 면접에서 "라이브 데모" 가능
