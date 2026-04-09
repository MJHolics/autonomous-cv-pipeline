# Phase 4 — 설계 근거 문서

> 왜 CARLA인가, 어떻게 데이터를 수집했는가, 통합 평가에서 무엇을 배웠는가.
> Phase 1~3의 모든 설계 결정이 여기서 검증된다.

---

## 1. 시뮬레이터 선택: CARLA

### 후보 비교

| 시뮬레이터 | 장점 | 단점 | 선택 여부 |
|---|---|---|---|
| **CARLA 0.9.15** | 오픈소스, 센서 다양(카메라/LiDAR/Radar/Semantic), Python API | 무거움(7GB+), 학습 곡선 | ✅ 선택 |
| SUMO | 경량 교통 시뮬레이션 | 비전 센서 없음 | ❌ |
| AirSim | 드론/자동차 지원 | 개발 중단, CARLA 대비 생태계 작음 | ❌ |
| nuScenes (오프라인) | 실제 데이터, 벤치마크 | 데이터 수집 불가, 커스텀 시나리오 없음 | Phase 5 후보 |
| LGSVL | Tier IV 지원 | CARLA 대비 커뮤니티 작음 | ❌ |

**CARLA를 선택한 핵심 이유**:
1. **Semantic segmentation 카메라**: 픽셀 단위 GT → Phase 1 Seg 절대 평가 가능
2. **Depth 카메라**: 절대 미터 GT → Phase 1 Depth 절대 평가 가능
3. **GT bbox 자동 생성**: 모든 NPC 위치를 World 좌표로 알 수 있음 → Detection/Tracking GT
4. **취업 목표 회사 요구사항과 직접 일치**: "CARLA/Omniverse 환경에서 데이터 생성·테스트"

### CARLA Python API 환경 설계

```
문제: Python 3.13 (Anaconda base) → carla wheel 없음
해결: Python 3.10 전용 conda 환경(carla_env) 생성
이유: CARLA 0.9.15 공식 wheel이 cp310(Python 3.10) 전용

carla_env:
  Python 3.10.20
  PyTorch 2.6.0+cu124  (GPU 가속 유지)
  CARLA 0.9.15 API
  Ultralytics, Transformers, OpenCV (Phase 1~3 모델 실행 가능)
```

Phase 1~3 모든 모델이 `carla_env` 단일 환경에서 실행되도록 설계.
이유: 통합 평가 노트북에서 환경 전환 없이 Detection→Seg→Depth→Tracking 연속 실행.

---

## 2. 데이터 수집 설계: carla_data_collector.py

### 수집 센서 구성

| 센서 | 역할 | Phase 연결 |
|---|---|---|
| RGB 카메라 | 입력 이미지 | Phase 1 Detection/Seg, Phase 2 Tracking/Pose |
| Depth 카메라 | 절대 깊이 GT (미터) | Phase 1 Depth 절대 평가 |
| Semantic Segmentation 카메라 | 픽셀 단위 class GT | Phase 1 Seg 절대 평가 |
| World 좌표 기반 GT bbox | 3D → 2D 투영 | Phase 1 Detection, Phase 2 Tracking |

**설계 결정**: 4개 센서를 동기화(synchronous mode)로 수집.
비동기 수집 시 프레임 불일치 → GT와 이미지가 다른 시점 → 평가 오류.

```python
world.apply_settings(carla.WorldSettings(
    synchronous_mode=True,
    fixed_delta_seconds=0.05  # 20 FPS 고정
))
```

### GT bbox 생성 방식

```
NPC 차량/보행자 World 좌표 (x,y,z)
    ↓ Camera extrinsic (E)
Camera 좌표 (Xc, Yc, Zc)
    ↓ Camera intrinsic (K)
Image 좌표 (u, v)
    ↓ 3D bbox 8 꼭짓점 투영
2D bbox (x1, y1, x2, y2)
```

이 투영 파이프라인이 Phase 3 BEV IPM의 역과정.
Phase 3에서 직접 구현한 K, E 행렬이 여기서 실제로 쓰인다.

### Semantic 카메라 인코딩 주의사항

CARLA semantic PNG 저장 포맷:
- R채널: semantic class ID (0~28)
- G채널: tag ID
- B채널: 0

**버그 이력 (2026-03-29)**:
`cv2.IMREAD_GRAYSCALE`로 읽으면 `0.114R + 0.587G + 0.299B` 가중 평균 → class ID 손실.
→ 반드시 BGR 읽기 후 `[:,:,2]` R채널 추출.

---

## 3. 통합 평가 설계

### 평가 파이프라인 순서
```
CARLA 데이터 로드
  ↓
[Detection] YOLOv8 + GT bbox → Precision/Recall/F1
  ↓
[Depth] DepthAnythingV2 + Depth GT → RMSE/AbsRel/δ1
  ↓
[Segmentation] SAM2 + Semantic GT → mIoU
  ↓
[Tracking] ByteTrack + 연속 프레임 → MOTA
  ↓
Phase 1~3 시뮬 결과와 비교 대시보드
```

### Phase 4 실행 결과 및 설계 검증 (2026-03-29)

| 모듈 | Phase 결과(시뮬) | Phase 4(CARLA) | 원인 분석 |
|---|---|---|---|
| Detection | mAP@50 ~0.6x | Precision 0.1328 | COCO128 도메인 갭 |
| Depth RMSE | 0.16m (합성GT) | 4.83m | 스케일 미정렬 (AbsRel 0.38로 실제 성능은 유사) |
| Segmentation | mIoU 0.694 (상대) | 0.0000 → **버그** | GRAYSCALE 읽기 오류 (수정 완료) |
| Tracking | MOTA 0.9412 | MOTA 0.2500 | Detection FN이 병목 |

### 설계에서 배운 것

**1. 평가 지표 선택의 중요성**
- RMSE만 보면 Depth가 4.83m로 완전히 실패한 것처럼 보임
- 하지만 AbsRel(0.38), δ1(0.42)는 Phase 1과 거의 동일
- **교훈**: 스케일 의존 지표(RMSE)와 스케일 불변 지표(AbsRel)를 함께 봐야 함

**2. 도메인 갭은 측정해야 안다**
- Phase 1에서 COCO128 mAP가 좋다고 CARLA에서도 좋을 것이라 가정했지만
- 실제로는 Precision 0.53(car 단독) — 게임엔진 렌더링과 실사의 차이
- **교훈**: 평가 환경을 바꾸면 성능도 바뀐다. Phase 4 같은 도메인 테스트가 필수

**3. 버그는 설계 단계의 가정에서 나온다**
- Seg mIoU=0.0000 — 코드는 맞았지만 CARLA 포맷 가정이 틀렸음
- "semantic은 grayscale 이미지일 것"이라는 무의식적 가정
- **교훈**: 외부 시스템(CARLA) 데이터를 받을 때는 포맷을 명시적으로 확인

**4. 트래킹 성능의 병목은 Detector**
- ByteTrack ID_SW=0 유지 → 알고리즘 자체는 정상
- MOTA 저하 원인은 FN(검출 실패) — Detection 품질 개선이 트래킹 개선의 전제

---

## 4. 개선 방향 (Phase 4 심화)

### 단기 (즉시 가능)
1. **Seg 재실행** — GRAYSCALE 버그 수정 완료 → 재실행으로 실제 mIoU 측정
2. **Depth 스케일 정렬** — `scale = median(gt)/median(pred)` 후 RMSE 재계산

### 중기 (선택)
3. **Detection CARLA 파인튜닝** — CARLA GT bbox로 YOLOv8 재학습 → Precision 0.8+ 목표
4. **FastAPI + Docker 대시보드** — Phase 1~4 결과를 웹 UI로 시각화 (기존 강점 활용)

### 장기 (취업 목표 연결)
5. **LiDAR 추가** — CARLA LiDAR 센서 + open3d (conda Python 3.11 환경)
6. **BEVFusion 통합** — Phase 3 IPM → LSS → Camera+LiDAR BEVFusion
7. **nuScenes 벤치마크** — 실제 공개 데이터셋으로 Phase 1~3 모델 재평가
