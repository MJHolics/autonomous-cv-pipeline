# 이력서 / 포트폴리오 문구 모음

> 각 항목은 **동사 + 기술 + 수치** 구조로 작성
> 면접관이 5초 안에 임팩트를 파악할 수 있도록

---

## 프로젝트 제목 (한 줄 소개)

**CARLA 기반 자율주행 멀티태스크 Perception 파이프라인**  
*Detection · Segmentation · Depth · Tracking · BEV · VLM · 추론 최적화 end-to-end 구현*

---

## 핵심 성과 (상단 강조용 — 3~5개 선택)

- CARLA 시뮬레이터 기반 합성 데이터로 YOLOv8 파인튜닝 → mAP@50 **0.43 → 0.68** (+59%)
- DepthAnything V2 스케일 정렬 파이프라인 구현 → RMSE **5.44 → 2.71m** (-50%)
- ByteTrack 직접 구현 (Kalman Filter + 이중 매칭) → MOT17 MOTA **0.9412**, ID_SW = 0
- Qwen2-VL-2B QLoRA 파인튜닝으로 Driving VQA ROUGE-L **0.027 → 0.759** (+28배)
- YOLOv8 TensorRT FP16 최적화 → 추론 속도 **132 → 316 FPS** (2.4배)

---

## Phase별 상세 문구

### Detection
- COCO128 데이터셋으로 YOLOv8n 학습 파이프라인 구축, mAP@50 = 0.99 달성
- FP/FN 유형 분류(배경 오검출 / 놓친 소형 객체)와 PR Curve threshold sweep으로 서비스 요구사항별 최적 임계값 도출
- CARLA 합성 도메인 파인튜닝으로 mAP@50 **0.43 → 0.68** (+59%), Tracking MOTA -0.69 → +0.29 연쇄 개선 확인

### Segmentation
- SAM2 Point/Box Prompt 정량 비교 (mIoU = 0.694), Prompt 방식 차이 원인을 GT 면적 분포로 분석
- SegFormer-B2 CARLA semantic segmentation 파인튜닝, mIoU **0.107 → 0.586** (+448%)
- median frequency balancing으로 클래스 불균형 보정, sky/road/vehicle 각 클래스 개선 폭 정량 측정

### Depth
- DepthAnything V2 상대 깊이 출력을 GT(미터)와 비교하기 위한 스케일 정렬 파이프라인 구현
- median scaling → disparity-space 정렬 2단계 개선으로 RMSE **5.44 → 2.71m** (-50%), delta1 개선
- 정렬 방식(metric vs disparity space) 차이를 수식과 실험으로 비교 분석

### Tracking
- ByteTrack 논문(ECCV 2022) 직접 구현 — Kalman Filter 상태 벡터 설계, IoU 헝가리안 매칭, 트랙 상태 전이
- 고신뢰도 1차 매칭 + 저신뢰도 2차 매칭 파이프라인으로 가려진 객체 ID 유지 (ID_SW = 0)
- MOT17 평가 결과 MOTA **0.9412** 달성

### Pose
- YOLOv8-pose 모델로 COCO Keypoints 기반 OKS(Object Keypoint Similarity) 평가 파이프라인 구축
- 관절별 OKS 분포 분석으로 손목/발목 등 원위 관절 오차 원인 정량 확인

### BEV / 3D Perception
- 핀홀 카메라 모델 + Homography로 IPM(Inverse Perspective Mapping) 직접 구현
- Lift-Splat-Shoot(NeurIPS 2020) 논문 구현 — frustum lifting + voxel pooling + BEV decoder
- PointNet++(NeurIPS 2017), PointPillars(CVPR 2019) 논문 직접 구현, KITTI 3D Detection 평가
- RGB → BEV Occupancy Grid 파이프라인: 역투영 GT 자동 생성 + ResNet-18 BEV U-Net, Binary mIoU **0.6882**

### VLM
- CARLA 500프레임에서 QA 7종 자동 생성 (위험도 판단 / 장면 설명 / 거리 추정 등) ShareGPT 포맷 출력
- Qwen2-VL-2B QLoRA(4bit) 파인튜닝, ROUGE-L **0.027 → 0.759** (+28배), BLEU-4 **0.004 → 0.546**
- ROUGE-L을 LCS DP로 직접 구현 (라이브러리 한국어 토큰화 버그 우회)
- FastAPI + Uvicorn 추론 서버 구현, Swagger UI 자동 문서화

### 추론 최적화 / 도메인 적응
- PyTorch → ONNX → TensorRT FP16 변환 파이프라인, YOLOv8 **132 → 316 FPS** (2.4배) 속도 향상
- Domain Randomization vs Pseudo-labeling 비교 실험: 소량 데이터 환경에서 DR은 모델 붕괴, PL이 신뢰도 **+24.5%** 효과 확인
- Knowledge Distillation(Hinton 2015 + FitNets) 직접 구현, YOLOv8l→n 지식 이식 mAP@50:95 **+0.0098**
- RAFT(ECCV 2020) Optical Flow + Semantic mask 결합으로 **detection 없이** 움직이는 차량/보행자 탐지
- UFLD(ECCV 2020) row-anchor classification 직접 구현, TuSimple Acc **1.0000** / FP **0.0** / FN **0.0**, **221 FPS** (4.52ms) 달성
- AB3DMOT(IROS 2020) 3D Kalman Filter(9D 상태벡터) + Sutherland-Hodgman BEV 회전 IoU 직접 구현, MOTA **0.8605** / MOTP **0.7222** / ID_SW **0**
- ResNet-18+FPN 단일 backbone → Detection+Depth+Segmentation 3개 태스크 동시 학습, GradNorm(ICML 2018) 자동 가중치 균형 구현 및 데이터 복잡도별 negative transfer 실증 분석
- PSMNet(CVPR 2018) lite 직접 구현 — 4D Cost Volume + Soft Argmin disparity regression, Classical(SAD) EPE 6.081px 대비 **EPE 1.406px** (4.3배↓), Depth MAE **2.797m** (LiDAR 없이 metric depth)
- PanopticFPN 직접 구현 — LightResNet+FPN+Semantic/Instance Head+Panoptic Merge, **PQ 0.7499** / mIoU **0.9245**; things(car 0.288)와 stuff(sky 1.000) 격차를 RQ 분해로 분석 → Mask2Former 필요성 실증
- DETR(ECCV 2020) 직접 구현 — Object Query + Hungarian Bipartite Matching Loss + GIoU Loss, **NMS 없이** anchor-free 검출; YOLO 계열과 수렴 속도·소형 객체 약점 정량 비교

### 인프라 / 배포
- nuScenes 공식 제출 포맷(results_nusc.json) 변환 + nuscenes-devkit NDS/mAP/mATE/mASE/mAOE 평가 파이프라인 구현
- FastAPI 마이크로서비스 5개(Detection/Depth/Segmentation/VLM/Gateway) + React+Vite 프론트엔드
- `docker compose up --build` 단일 명령으로 전체 서비스 실행, asyncio.gather 병렬 추론으로 응답 지연 최소화

---

## 한 줄 요약 버전 (LinkedIn / 자기소개)

CARLA 시뮬레이터 기반 자율주행 Perception 파이프라인 프로젝트. Detection(mAP +59%) · Depth(RMSE -50%) · 3D Tracking(MOTA 0.86) · VLM(ROUGE-L +28배) · Stereo Depth(EPE 1.406px) · Panoptic Seg(PQ 0.75) · DETR(anchor-free) 등 10개 분야 논문을 직접 구현·평가하고, Domain Randomization 실패 분석 / GradNorm negative transfer 실증 등 부정적 결과까지 정량 분석해 마이크로서비스 대시보드로 통합했습니다.

---

## 기술 키워드 (JD 매칭용)

```
PyTorch · CUDA · TensorRT · ONNX · YOLOv8 · SegFormer · SAM2 · DepthAnything V2
ByteTrack · Kalman Filter · Lift-Splat-Shoot · PointPillars · BEV · Occupancy Grid
RAFT Optical Flow · Knowledge Distillation · QLoRA · Qwen2-VL · CARLA · nuScenes
FastAPI · Docker · React · Vite · asyncio · COCO · MOT17 · KITTI · NYU Depth v2
```

---

## 회사별 강조 포인트 (자율주행 JD 분석 기준)

| 회사 유형 | 강조할 항목 |
|-----------|------------|
| 자율주행 스타트업 | CARLA 파이프라인, BEV, Occupancy Grid, nuScenes 리더보드 |
| 반도체 / 엣지 AI | TensorRT FP16, KD, ONNX 최적화, FPS 수치 |
| 로보틱스 | Depth + BEV + Optical Flow, CARLA synchronous mode |
| AI 플랫폼 | VLM QLoRA, FastAPI 마이크로서비스, Docker, asyncio |
