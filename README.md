# Autonomous CV Pipeline

> 자율주행/로보틱스 취업을 위한 CV 스킬업 프로젝트
> CARLA 시뮬레이터 기반 멀티태스크 Perception 파이프라인 + VLM 파인튜닝 + TensorRT 최적화 + FastAPI 배포

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0+cu124-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.4-green)
![TensorRT](https://img.shields.io/badge/TensorRT-FP16-red)

---

## 핵심 성과 수치

| 분야 | 작업 | 모델 | 결과 |
|------|------|------|------|
| Detection | CARLA 도메인 적응 | YOLOv8 파인튜닝 | mAP@50 **0.43 → 0.68** (+59%) |
| Segmentation | CARLA 시맨틱 | SegFormer-B2 파인튜닝 | mIoU **0.107 → 0.586** (+448%) |
| Depth | 스케일 정렬 | DepthAnything V2 | RMSE **5.44 → 2.71m** (-50%) |
| Tracking | MOT17 평가 | ByteTrack (직접 구현) | MOTA = **0.9412** |
| VLM | Driving VQA | Qwen2-VL-2B QLoRA | ROUGE-L **0.027 → 0.759** (+28배) |
| 추론 최적화 | TensorRT FP16 | YOLOv8 | **132 → 316 FPS** (2.4x) |
| 도메인 적응 | Pseudo-labeling | YOLOv8 | 실제 도로 신뢰도 **+24.5%** (레이블 없이) |
| 리더보드 제출 | nuScenes 평가 | LSS BEV | NDS/mAP/mATE/mASE/mAOE **공식 파이프라인** |
| BEV Occupancy | 카메라→점유격자 | ResNet-18 + BEV U-Net | Binary mIoU **0.6882** |
| Knowledge Distillation | Teacher→Student 압축 | YOLOv8l→n KD | mAP@50:95 **0.3501→0.3599** (+3%) |
| Optical Flow | 움직이는 객체 탐지 | RAFT (ECCV 2020) | Flow+Semantic 결합 **detection 없이 차량/보행자 탐지** |
| Lane Detection | Row-anchor 분류 | UFLD (ECCV 2020) | Acc **1.0000**, FP **0.0000**, FN **0.0000**, **221 FPS** (4.52ms) |
| 3D MOT | 3D Kalman + 헝가리안 | AB3DMOT (IROS 2020) | MOTA **0.8605**, MOTP **0.7222**, ID_SW = **0** |
| Multi-Task Learning | Det+Depth+Seg 동시 | ResNet-18 + FPN + GradNorm | GradNorm vs Equal weighting 비교 — 단순 합성 데이터에서 Equal 수렴 우위 (negative transfer 실증) |
| Stereo Depth | LiDAR 없이 metric depth | PSMNet-lite (4D Cost Volume) | EPE **1.406px** (Block Matching 6.081 대비 **4.3배↓**), D1 **12.32%**, Depth MAE **2.797m** |
| Panoptic Seg | stuff + things 통합 | PanopticFPN (ResNet+FPN) | PQ **0.7499**, mIoU **0.9245** (sky PQ=1.000 / car PQ=0.288 차이 원인 분석) |
| DETR | Anchor-free Detection | DETR (ECCV 2020) | Hungarian Matching + GIoU Loss + Object Query, **NMS 불필요** |

---

## 빠른 시작

**VLM FastAPI 서버:**
```bash
cd vlm_driving
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://127.0.0.1:8000/docs
```

**Dashboard (마이크로서비스 전체):**
```bash
cd dashboard
docker compose up --build
# Frontend: http://localhost:3000
```

---

## 프로젝트 구조

```
autonomous_cv_pipeline/
├── phase1_basics/
│   ├── detection/          # YOLOv8 커스텀 학습 + mAP 평가 파이프라인
│   ├── segmentation/       # SAM2 Point/Box Prompt 비교 평가
│   └── depth/              # DepthAnything V2 + RMSE/delta1 평가
├── phase2_tracking_pose/
│   ├── tracking/           # ByteTrack 직접 구현 (Kalman Filter + 이중 매칭)
│   └── pose/               # YOLOv8-pose + OKS 평가 파이프라인
├── phase3_bev/
│   ├── 01_bev_ipm_pipeline.ipynb          # IPM + Homography 직접 구현
│   ├── 03_nuscenes_multicamera_bev.ipynb  # Lift-Splat-Shoot (논문 구현)
│   └── pointcloud/
│       ├── 01_pointnet_plus_plus.ipynb    # PointNet++ 직접 구현
│       └── 02_kitti_3d_detection.ipynb    # PointPillars (논문 구현)
├── phase4_carla/
│   ├── data_collection/    # CARLA 자동 데이터 수집 (synchronous mode)
│   ├── evaluation/         # Phase 1~3 통합 평가 파이프라인
│   └── finetuning/         # YOLOv8 / SegFormer / Depth 도메인 적응
├── vlm_driving/
│   ├── app/main.py         # FastAPI 서빙 앱
│   ├── Dockerfile
│   └── 01~04 notebooks     # 데이터 파이프라인 -> 파인튜닝 -> 평가 -> 배포
├── skillup_round4/
│   ├── tensorrt_optimization/  # PyTorch -> ONNX -> TensorRT FP16 벤치마크
│   ├── synthetic_to_real/      # Domain Randomization + Pseudo-labeling 실험
│   └── nuscenes_submission/    # nuScenes 공식 제출 포맷 + NDS/mAP 평가 파이프라인
├── skillup_round5/
│   ├── occupancy_grid/         # CARLA RGB -> BEV Occupancy Grid (ResNet-18 + BEV U-Net)
│   ├── knowledge_distillation/ # YOLOv8l(Teacher) -> YOLOv8n(Student) KD 구현
│   └── optical_flow/           # RAFT Optical Flow + Semantic 결합 움직임 탐지
├── skillup_round6/
│   ├── lane_detection/         # UFLD Row-anchor 분류 직접 구현 + TuSimple 메트릭
│   ├── 3d_tracking/            # AB3DMOT 3D Kalman Filter + BEV 회전 IoU + MOTA/MOTP
│   └── multitask_learning/     # ResNet+FPN 단일 backbone → Det+Depth+Seg + GradNorm
└── dashboard/
    ├── docker-compose.yml  # 마이크로서비스 통합 (5 서비스 + frontend)
    └── frontend/           # React + Vite + TypeScript
```

---

## Phase 1 — CV 기초 파이프라인

**Detection** `phase1_basics/detection/`
- YOLOv8n COCO128 학습. mAP@50 = **0.99**
- FP/FN 유형 분석, Confusion Matrix, PR Curve 직접 구현
- Threshold sweep으로 서비스 요구사항별 최적 임계값 도출

**Segmentation** `phase1_basics/segmentation/`
- SAM2 Point Prompt vs Box Prompt 정량 비교. mIoU = **0.694**
- Point Prompt가 평균 6% 낮은 원인을 GT 면적 분포로 분석

**Depth** `phase1_basics/depth/`
- DepthAnything V2 Small. NYU Depth v2 GT 비교
- 스케일 불변 정렬(median scaling) 후 RMSE = **0.16m**, delta1 = 0.41

---

## Phase 2 — Tracking + Pose

**ByteTrack 직접 구현** `phase2_tracking_pose/tracking/`
- Kalman Filter, IoU 기반 이중 매칭, 저신뢰도 검출 보조 트랙 직접 구현
- MOT17 서브셋 평가: MOTA = **0.9412**, ID_SW = 0

**Pose Estimation** `phase2_tracking_pose/pose/`
- YOLOv8-pose 17 keypoint 검출
- OKS(Object Keypoint Similarity) 평가 파이프라인 직접 구현

---

## Phase 3 — BEV + 3D Perception

| 노트북 | 논문 | 핵심 구현 |
|--------|------|-----------|
| `01_bev_ipm_pipeline.ipynb` | IPM | 핀홀 카메라 K/E 행렬, Homography 직접 계산 |
| `03_nuscenes_multicamera_bev.ipynb` | Lift-Splat-Shoot (NeurIPS 2020) | Lift → Splat(scatter_add) → Shoot(U-Net) |
| `pointcloud/01_pointnet_plus_plus.ipynb` | PointNet++ (NeurIPS 2017) | Set Abstraction, FPS, Ball Query 직접 구현 |
| `pointcloud/02_kitti_3d_detection.ipynb` | PointPillars (CVPR 2019) | Pillar Feature Net, Scatter, SSD Head |

---

## Phase 4 — CARLA 통합 평가 + 도메인 적응

**데이터 수집**: CARLA 0.9.15 synchronous mode, RGB/Depth/Semantic GT 자동 생성

**도메인 적응 결과:**

| 모델 | 지표 | 적응 전 | 적응 후 | 개선 |
|------|------|---------|---------|------|
| YOLOv8 | mAP@50 | 0.43 | **0.68** | +59% |
| YOLOv8 | MOTA | -0.69 | **+0.29** | Detection이 Tracking 병목 확인 |
| SegFormer-B2 | mIoU | 0.107 | **0.586** | +448% |
| DepthAnything V2 | RMSE | 5.44m | **2.71m** | -50% |

---

## VLM — Autonomous Driving VQA

**Qwen2-VL-2B QLoRA 파인튜닝** (4-bit, RTX 4080 SUPER 16GB)

- 학습 데이터: CARLA GT에서 7가지 QA 자동 생성 (2,243개)
- QA 유형: 장면설명 / 위험평가 / 행동권고 / 보행자 / 최근접객체 / 수량 / 안전거리

| 지표 | Base 모델 | 파인튜닝 후 | 향상 |
|------|-----------|-------------|------|
| BLEU-4 | 0.004 | 0.546 | +136배 |
| ROUGE-L | 0.027 | 0.759 | +28배 |
| 거리 MAE | — | 5.36m | 68% 가 5m 이내 |

**API 응답 예시 (`/predict/batch`):**
```json
{
  "scene_description": "도시 도로 주행 장면. 좌측 13.0m 거리에 보행자.",
  "danger_assessment": "예. 좌측 13.0m 거리에 보행자가 있어 속도를 줄이고 주의 필요.",
  "action_recommendation": "현재 속도를 유지하되 좌측의 보행자를 모니터링하세요.",
  "pedestrian_check": "예. 좌측 13.0m 거리에 보행자가 1명 있습니다.",
  "nearest_object": "좌측 보행자, 약 13.0m 거리.",
  "object_count": "차량과 보행자 총 2개 감지.",
  "safety_distance": "전방 차량과의 안전 거리(19.5m) 확보됨.",
  "elapsed_sec": 22.3
}
```

---

## Skillup Round 4 — 추론 최적화 + 도메인 적응 + 리더보드

**YOLOv8 TensorRT FP16 최적화** `skillup_round4/tensorrt_optimization/`

| 포맷 | FPS | PyTorch 대비 |
|------|-----|-------------|
| PyTorch FP32 (기준) | 132 | 1.0x |
| PyTorch FP16 | 127 | 0.96x |
| ONNX Runtime CPU | 15 | 0.11x |
| ONNX Runtime CUDA | 123 | 0.93x |
| **TensorRT FP16** | **316** | **2.4x** |

- 레이어 퓨전(Conv+BN+ReLU 단일 커널) + 텐서코어 활용
- 실시간 기준 30 FPS 대비 10x 이상 달성
- 검출 정확도 동일 유지 확인

---

**Synthetic-to-Real 도메인 적응** `skillup_round4/synthetic_to_real/`

CARLA 합성 데이터로 학습한 모델이 실제 도로에서 겪는 도메인 갭을 정량 측정하고 두 가지 기법을 비교 실험.

| 기법 | 신뢰도 변화 | 검출률 변화 | 결론 |
|------|------------|------------|------|
| CARLA 원본 (기준) | 0.437 | 45.0% | 합성→실제 갭 확인 |
| Domain Randomization | 0.437 → 0.305 (-30%) | 45% → 1% | **소량 데이터 재학습에 부적합** |
| **Pseudo-labeling** | 0.437 → **0.544 (+24.5%)** | 45% → 20% | **레이블 없는 실제 데이터 활용 유효** |

- DR 실패 원인 분석: 수백장 수준 소량 CARLA 데이터에서 재학습 기반 DR은 기존 feature 파괴
- DR은 from-scratch 학습에 적합, 소량 fine-tuning 환경에서는 Pseudo-labeling이 우월
- COCO val2017 실제 도로 이미지 100장으로 비지도(unsupervised) 평가

---

**nuScenes 리더보드 제출 파이프라인** `skillup_round4/nuscenes_submission/`

LSS BEV 모델 출력을 nuScenes 공식 제출 포맷으로 변환하고, nuscenes-devkit으로 공식 메트릭을 계산하는 엔드투엔드 파이프라인 구현.

```
[모델 출력] → [DetectionBox 변환] → [results_nusc.json] → [accumulate] → [NDS / mAP / mATE / mASE / mAOE]
```

- 10개 클래스(car / truck / bus / pedestrian / bicycle 등) 전체 스키마 구현
- `nuscenes-devkit` `accumulate` → `calc_ap` → `calc_tp` → NDS 직접 계산
- SOTA 비교: BEVFusion(NDS 0.713), CenterPoint(0.598), BEVDet(0.482), FCOS3D(0.428)
- 실제 nuScenes mini 연동 가이드 포함 (데이터 확보 시 동일 파이프라인으로 즉시 실제 NDS 산출)

---

## Skillup Round 5 — BEV Occupancy / Knowledge Distillation / Optical Flow

**BEV Occupancy Grid** `skillup_round5/occupancy_grid/`

CARLA RGB 단일 이미지로 BEV 점유 격자(Occupancy Grid)를 예측. Tesla FSD, UniAD, OccNet 방향의 camera-only 3D 공간 이해.

```
RGB (256×256) → ResNet-18 인코더 → BEV U-Net 디코더 → 5-class Occupancy Map (100×100)
GT: CARLA depth + semantic → 역투영(backprojection) → BEV 격자 투영
```

| 클래스 | IoU |
|--------|-----|
| free | 0.7750 |
| obstacle | 0.5984 |
| road / vehicle / pedestrian | 0.0000 (GT < 0.5%, 데이터 병목) |
| **Binary mIoU** (free vs occupied) | **0.6882** |

- median frequency balancing으로 클래스 불균형 보정
- road/vehicle 클래스 학습 실패 원인 정량 분석: Town01 단일 장면에서 BEV 내 GT 비율 0.1~0.5%로 극히 희귀

---

**Knowledge Distillation** `skillup_round5/knowledge_distillation/`

YOLOv8l(Teacher, 43.7M) → YOLOv8n(Student, 3.2M) 지식 이식. Hinton 2015 + FitNets 직접 구현.

| 모델 | mAP@50 | mAP@50:95 | 파라미터 |
|------|--------|-----------|---------|
| Baseline YOLOv8n | 0.5985 | 0.3501 | 3.2M |
| KD Student YOLOv8n | 0.5924 | **0.3599** | 3.2M |

- Response-based KD: Temperature scaling + KL divergence (T=4 최적)
- Feature-based KD: Forward hook으로 backbone 중간 레이어 MSE loss
- mAP@50:95 기준 +0.0098 개선 (같은 크기에서 지식 이식 효과)

---

**Optical Flow** `skillup_round5/optical_flow/`

RAFT(ECCV 2020 Best Paper) 적용 및 Classical 기법과 정량 비교.

| 지표 | RAFT | Farneback |
|------|------|-----------|
| EPE (Pseudo-GT) | 427.0px | 79.2px |

- RAFT vs Farneback EPE 역전 원인 분석: CARLA 합성 프레임 pseudo-GT가 smoothed flow에 편향
- Flow magnitude + Semantic mask 결합 → detection 없이 움직이는 차량/보행자 픽셀 추출
- 실제 Sintel/KITTI GT 기준에서는 RAFT SOTA (EPE 1.6px vs Farneback 5.8px)

---

## Skillup Round 6 — Lane Detection / 3D MOT / Multi-Task Learning

**Lane Detection (UFLD)** `skillup_round6/lane_detection/`

- UFLD(ECCV 2020) row-anchor classification 직접 구현 — 세그멘테이션 대비 18배 빠름
- Structural Loss: 인접 row anchor 연속성 제약 (기하학적 사전지식 주입)
- TuSimple Acc **1.0000** / FP **0.0000** / FN **0.0000** / **221 FPS** (4.52ms)

**3D Multi-Object Tracking (AB3DMOT)** `skillup_round6/3d_tracking/`

- 3D Kalman Filter (9D 상태벡터: 위치+방향+크기+속도) 직접 구현
- BEV 회전 IoU: Sutherland-Hodgman 폴리곤 클리핑 알고리즘
- MOTA **0.8605** / MOTP **0.7222** (3D IoU) / ID_SW = **0**

**Multi-Task Learning (GradNorm)** `skillup_round6/multitask_learning/`

- ResNet-18 + FPN 단일 backbone → Detection + Depth + Segmentation 동시 학습
- GradNorm (ICML 2018): gradient norm 기반 자동 가중치 균형
- 단순 합성 데이터에서 Equal weighting 수렴 우위 확인 → 알고리즘 효과는 태스크 이질성에 달려 있음 (negative transfer 실증)

---

## Skillup Round 7 — Stereo Depth / Panoptic Seg / DETR

**Stereo Depth** `skillup_round7/stereo_depth/`

- PSMNet-lite 직접 구현: FeatureNet + 4D Cost Volume + Soft Argmin disparity regression
- 스테레오 삼각측량: `depth = f×B/d` — LiDAR 없이 metric depth 추정
- Classical (Block Matching SAD) vs Learning-based 정량 비교

| 방법 | EPE | D1 | 3PE |
|------|-----|----|-----|
| Block Matching (Classical) | 6.081 px | 15.28% | 15.29% |
| PSMNet-lite (Learning) | **1.406 px** | **12.32%** | **12.79%** |

- Depth MAE **2.797m** (metric depth, LiDAR 없이 스테레오만으로)

**Panoptic Segmentation** `skillup_round7/panoptic_seg/`

- PanopticFPN 직접 구현: LightResNet + FPN + Semantic Head + Instance Head + Panoptic Merge
- Panoptic Quality(PQ) = SQ × RQ 분해 평가 파이프라인 구현
- Hungarian 매칭으로 things 인스턴스 최적 할당, Mask2Former와의 구조적 차이 비교

| 클래스 | 타입 | PQ | SQ | RQ |
|--------|------|----|----|----|
| road | stuff | 0.9809 | 0.9809 | 1.000 |
| sky | stuff | 1.0000 | 1.0000 | 1.000 |
| building | stuff | 0.9996 | 0.9996 | 1.000 |
| car | things | 0.2880 | 0.3662 | 0.3797 |
| pedestrian | things | 0.4812 | 0.5368 | 0.5557 |
| **전체** | | **0.7499** | | |

- things(car/pedestrian) 낮은 원인: 인스턴스 겹침 + Instance Head 마스크 경계 부정확 → Mask2Former 필요성 실증

**DETR** `skillup_round7/detr/`

- DETR(ECCV 2020) 핵심 구현: CNN backbone + Transformer Enc-Dec + Object Query
- Hungarian Matching Loss + GIoU Loss + 2D Sinusoidal Positional Encoding 직접 구현
- YOLO(anchor-based+NMS) vs DETR(anchor-free, NMS 불필요) 구조적 비교
- 단점 분석: 학습 수렴 느림(500ep), 소형 객체 약함 → Deformable DETR 해결 방향 제시

---

## Dashboard — 마이크로서비스

| 서비스 | 포트 | 역할 |
|--------|------|------|
| gateway | 8000 | asyncio.gather() 병렬 라우터 |
| detection | 8001 | YOLOv8 (CARLA 파인튜닝 best.pt) |
| depth | 8002 | DepthAnything V2 |
| segmentation | 8003 | SegFormer ADE20K |
| vlm | 8004 | Qwen2-VL-2B QLoRA |
| frontend | 3000 | React + Vite + TypeScript |

---

## 논문 구현 목록

| 논문 | 학회/연도 | 구현 위치 |
|------|-----------|-----------|
| ByteTrack | ECCV 2022 | `phase2_tracking_pose/tracking/` |
| DepthAnything V2 | 2024 | `phase1_basics/depth/` |
| Lift-Splat-Shoot | NeurIPS 2020 | `phase3_bev/03_nuscenes_multicamera_bev.ipynb` |
| PointNet++ | NeurIPS 2017 | `phase3_bev/pointcloud/01_pointnet_plus_plus.ipynb` |
| PointPillars | CVPR 2019 | `phase3_bev/pointcloud/02_kitti_3d_detection.ipynb` |
| Pseudo-labeling (Self-training) | — | `skillup_round4/synthetic_to_real/` |
| RAFT | ECCV 2020 | `skillup_round5/optical_flow/` |
| Knowledge Distillation (Hinton) | NeurIPS 2015 | `skillup_round5/knowledge_distillation/` |
| FitNets Feature Distillation | ICLR 2015 | `skillup_round5/knowledge_distillation/` |
| UFLD (Ultra-Fast Lane Detection) | ECCV 2020 | `skillup_round6/lane_detection/` |
| AB3DMOT (3D Multi-Object Tracking) | IROS 2020 | `skillup_round6/3d_tracking/` |
| GradNorm (Multi-Task Learning) | ICML 2018 | `skillup_round6/multitask_learning/` |
| PSMNet (Stereo Depth) | CVPR 2018 | `skillup_round7/stereo_depth/` |
| Panoptic Segmentation + PQ | CVPR 2019 | `skillup_round7/panoptic_seg/` |
| DETR | ECCV 2020 | `skillup_round7/detr/` |

---

## 주요 설계 결정

| 결정 | 이유 |
|------|------|
| CARLA synchronous mode | 비동기 모드는 센서 GT 타임스탬프 불일치 발생 |
| SegFormer vs SAM2 | SAM2=인스턴스 분리, CARLA GT=semantic — 태스크 미스매치 |
| median freq balancing | inverse freq는 sky 클래스 과가중치 문제 |
| disparity-space 정렬 | metric space 대비 비선형 오차 2배 감소 |
| QLoRA 4-bit | 7B급 모델을 16GB VRAM에서 학습하기 위한 최소 요건 |
| LCS DP 직접 구현 | rouge_score 라이브러리의 한국어 빈 배열 처리 버그 우회 |
| TensorRT FP16 | FP32 대비 2.4x 속도 향상, 검출 결과 동일 유지 확인 |
| DR보다 Pseudo-labeling | 소량 합성 데이터 환경에서 DR 재학습은 feature 파괴, PL이 실효적 |
| nuScenes 공식 파이프라인 | nuscenes-devkit accumulate→calc_ap→calc_tp→NDS 직접 계산으로 평가 독립성 확보 |

---

## 환경

```
OS      : Windows 11, CUDA 12.4
GPU     : RTX 4080 SUPER 16GB
Python  : Anaconda 3.12 (autonomous_cv 커널)
PyTorch : 2.6.0+cu124
패키지  : Ultralytics 8.3.167, transformers, supervision, timm,
          lapx, filterpy, onnxruntime-gpu, tensorrt
```
