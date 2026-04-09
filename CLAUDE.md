# Autonomous CV Pipeline — 프로젝트 컨텍스트

## 목표
자율주행/로보틱스 CV 엔지니어 취업을 위한 스킬업 프로젝트.
CARLA 시뮬레이터 기반 자율주행 멀티태스크 Perception 파이프라인 구축.

## 사용자 배경
- AI 엔지니어 부트캠프 수료 (Mission 10~18)
- **강점**: NLP/LLM (RAG, PEFT, Transformer 파인튜닝), Docker, FastAPI, 모델 최적화 (ONNX, 양자화)
- **CV 경험**: YOLO v5/8/11로 객체검출 (약제 73종, mAP@50=0.99), SAM/MiDaS를 도구로 사용
- **없는 것**: Tracking, Pose estimation, BEV, 3D perception, 시뮬레이션 파이프라인

## 취업 목표 회사 요구사항
- Omniverse / CARLA 환경에서 데이터 생성·테스트·평가 파이프라인 개발
- Detection, Segmentation, Tracking, Depth, Pose, BEV, 멀티뷰, 3D perception
- 논문/오픈소스 분석 후 내재화

## 환경
- Windows 11, Python 3.13.5 (venv)
- RTX 4080 SUPER 16GB, CUDA 12.4
- PyTorch 2.6.0+cu124, Ultralytics 8.3.167
- 설치완료: supervision, transformers, timm, lapx, filterpy, pycocotools
- open3d는 Python 3.13 미지원 → Phase 3 때 conda Python 3.11 환경 별도 생성 예정

## 4단계 로드맵

### Phase 1 — CV 기초 파이프라인 (현재 진행)
Detection / Segmentation / Depth를 "도구 사용"에서 "원리 이해 + 평가 파이프라인 구축"으로 심화
- `phase1_basics/detection/` : YOLOv8 커스텀 학습 + mAP 평가 파이프라인
- `phase1_basics/segmentation/` : SAM2 파인튜닝 + IoU 평가
- `phase1_basics/depth/` : DepthAnythingV2 + RMSE/δ1 평가
- 데이터셋: COCO128 (detection), NYU Depth v2 (depth)
- 핵심 논문: DepthAnything v2 (2024)

### Phase 2 — Tracking + Pose
- `phase2_tracking_pose/tracking/` : ByteTrack 구현
- `phase2_tracking_pose/pose/` : ViTPose 또는 MediaPipe
- 데이터셋: MOT17, COCO Keypoints
- 핵심 논문: ByteTrack (2022), ViTPose (2022)

### Phase 3 — BEV + 멀티뷰
- `phase3_bev/` : IPM 직접 구현, nuScenes 데이터셋
- 핵심 논문: BEVFusion (2022), BEVDet (2022)

### Phase 4 — CARLA 시뮬레이션 파이프라인
- `phase4_carla/data_collection/` : CARLA 자동 데이터 수집 스크립트
- `phase4_carla/evaluation/` : Phase 1~3 모델 통합 자동 평가
- FastAPI + Docker 대시보드 (기존 강점 활용)

## 진행 상황
- [x] 프로젝트 구조 생성
- [x] 환경 세팅 (패키지 설치)
- [x] Phase 1 Detection 노트북 작성 (`phase1_basics/detection/01_yolov8_detection_pipeline.ipynb`)
- [ ] Phase 1 Segmentation 노트북 ← **다음 작업**
- [ ] Phase 1 Depth 노트북

## 환경 주의사항
- **venv는 CPU-only (torch 2.11.0)** — 사용하지 말 것
- **Anaconda Python** 사용: `C:\Users\apple\Anaconda3` (PyTorch 2.6.0+cu124, CUDA 정상)

## 작업 원칙
- 단순 API 호출이 아니라 **원리를 코드로 직접 구현**하는 방향
- 각 Phase마다 핵심 논문 1~2개를 읽고 구현으로 연결
- 평가 파이프라인을 항상 함께 구축 (숫자로 성능을 증명)
- Docker 기반 배포까지 이어지는 구조 유지

## 노트북 작성 규칙 (필수)
- **한글 폰트**: 모든 노트북 첫 번째 셀에 반드시 포함
  ```python
  import matplotlib
  matplotlib.rcParams['font.family'] = 'Malgun Gothic'
  matplotlib.rcParams['axes.unicode_minus'] = False
  ```
- **커널**: `autonomous_cv` 커널 고정 (노트북 metadata에 명시)
- **Ultralytics names**: `dataset_info['names']`는 dict → 슬라이스 시 `list(names.values())[:N]` 사용

## 문제 해결 로그 관리 (필수)
- 각 Phase 폴더에 `TROUBLESHOOTING.md` 파일 유지
  - Phase 1: `phase1_basics/TROUBLESHOOTING.md`
  - Phase 2: `phase2_tracking_pose/TROUBLESHOOTING.md`
  - Phase 3: `phase3_bev/TROUBLESHOOTING.md`
  - Phase 4: `phase4_carla/TROUBLESHOOTING.md`
  - Dashboard: `dashboard/TROUBLESHOOTING.md`
- 에러 발생 시 즉시 해당 파일에 누적 기록: 증상 / 원인 / 해결 / 교훈

## Dashboard 구조 (Phase 4 결과물)

`dashboard/` — `docker compose up --build` 하나로 실행

| 서비스 | 포트 | 역할 |
|---|---|---|
| gateway | 8000 | asyncio.gather() 병렬 라우터 |
| detection | 8001 | YOLOv8 (CARLA finetuned best.pt) |
| depth | 8002 | DepthAnythingV2 Metric-Outdoor |
| segmentation | 8003 | SegFormer ADE20K pretrained |
| vlm | 8004 | Qwen2-VL-2B QLoRA |
| frontend | 3000 | React + Vite + TypeScript → nginx |

### Dashboard 알려진 이슈 (이미 수정됨)
- `libgl1-mesa-glx` → Debian bookworm에서 사라짐. `libgl1` 사용
- `import.meta.env` TS 에러 → `tsconfig.json`에 `"types": ["vite/client"]` 필요
