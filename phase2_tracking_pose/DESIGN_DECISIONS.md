# Phase 2 — 설계 근거 문서

> 왜 ByteTrack을 직접 구현했는가, ViTPose 대신 YOLOv8-pose를 고른 이유는 무엇인가.
> 의사결정 과정을 기록해 이후 개선 방향의 근거로 삼는다.

---

## 1. Tracking — ByteTrack

### 목표 설정
"Tracking = Detection + Re-ID" 라는 공식 너머, **ByteTrack이 왜 ID Switch를 줄이는지**를
직접 구현하며 이해하는 것. MOTA 계산도 직접 구현.

### 모델/알고리즘 선정 이유: ByteTrack

| 후보 | 특징 | 장점 | 단점 | 선택 여부 |
|---|---|---|---|---|
| **ByteTrack** (2022) | Detection-only tracking | 구현 단순, Re-ID 불필요, SOTA급 | 가림에 약함 | ✅ 선택 |
| DeepSORT | Detection + Re-ID CNN | Re-ID로 가림 극복 | 별도 Re-ID 모델 필요 | ❌ |
| OC-SORT | ByteTrack 개선판 | 더 정확한 칼만 예측 | 복잡도 높음, Phase 2 오버스펙 | ❌ |
| SORT | ByteTrack 원조 | 매우 단순 | low-confidence 박스 버림 → ID_SW 많음 | ❌ |
| StrongSORT | 최신 SOTA | 높은 정확도 | 복잡, 원리 학습에 부적합 | ❌ |

**ByteTrack을 고른 핵심 이유**:
- 논문(2022, ECCV)이 명확하게 읽힘 — "low-confidence 박스를 버리지 않는다"는 한 문장이 전부
- Re-ID 없이 **IoU 매칭만으로** 높은 MOTA 달성 → 구현이 단순하고 원리가 명확
- lapx(LAP solver) 패키지가 설치되어 있어 헝가리안 알고리즘 바로 사용 가능

### ByteTrack 핵심 아이디어 구현

```
일반 Tracker:
  confidence > threshold → 매칭 시도 → 나머지 버림

ByteTrack:
  1단계: high-conf 박스 → IoU 매칭
  2단계: 1단계에서 매칭 안 된 트랙 → low-conf 박스와 재매칭
  → 가림이나 순간 신뢰도 하락 시에도 트랙 유지 → ID_SW 감소
```

이 2단계 매칭이 SORT 대비 ByteTrack의 핵심 차별점.
직접 구현하며 "왜 ID_SW=0이 나오는지" 체감할 수 있었음.

### 평가 지표: MOTA

```
MOTA = 1 - (FP + FN + ID_SW) / GT_total
```

MOTA를 선택한 이유:
- Multi-Object Tracking 표준 지표 (MOT Challenge 공식)
- FP/FN/ID_SW 세 요소를 한 번에 측정 — 어디서 성능이 빠지는지 진단 가능
- Phase 4 CARLA 결과(MOTA=0.25)와 비교 시 FN이 주요 원인임을 바로 파악 가능

### Phase 2 vs Phase 4 성능 격차 분석

```
Phase 2 (시뮬): MOTA=0.9412  — 단순 시나리오, GT Detection 사용
Phase 4 (CARLA): MOTA=0.2500 — 실 렌더링, Detection 모델 품질이 병목
```

**핵심 인사이트**: ByteTrack 알고리즘 자체는 CARLA에서도 ID_SW=0을 유지.
성능 저하는 알고리즘 문제가 아니라 **Detection 입력 품질 문제**.
→ Detection 재파인튜닝 → MOTA 자동 상승 예측 가능.

### 칼만 필터 설계 결정
상태 벡터: `[x, y, w, h, vx, vy, vw, vh]` (8차원)
- 위치(x,y)와 크기(w,h), 각각의 속도 포함
- filterpy 라이브러리 사용 — 직접 구현 대신 검증된 라이브러리 활용
- 이유: 칼만 필터 수식 자체보다 **트래킹 파이프라인 전체 흐름** 이해가 목적

---

## 2. Pose Estimation — YOLOv8-pose

### 목표 설정
Keypoint 검출의 **평가 지표(OKS)를 직접 구현**하고, 골격 시각화 + 보행자 의도 예측까지.
자율주행에서 Pose가 왜 필요한지(보행자가 길을 건너려는지 예측)를 실용적 맥락으로 연결.

### 모델 선정 이유: YOLOv8-pose

| 후보 | 특징 | 장점 | 단점 | 선택 여부 |
|---|---|---|---|---|
| **YOLOv8-pose** | Detection + Keypoint 동시 | 단일 모델, 빠른 추론, 17 keypoint | | ✅ 선택 |
| ViTPose (2022) | Transformer 기반 전용 Pose | 최고 정확도, 논문 수준 | 별도 Detector 필요, 설치 복잡 | 후보 유지 |
| MediaPipe Pose | Google, 경량 | 모바일 최적화, 33 keypoint | 커스텀 불가, 블랙박스 | ❌ |
| HRNet | High-Resolution 전용 | 높은 keypoint 정확도 | 느림, 별도 Detector 필요 | ❌ |

**YOLOv8-pose를 고른 이유**:
- Phase 2는 "Pose 파이프라인 구축"이 목표 — 최고 정확도보다 **속도 × 구현 용이성**
- Ultralytics 생태계로 Detection(Phase 1)과 동일한 환경에서 실행
- 17 keypoint(COCO format)가 자율주행 보행자 의도 예측에 충분

### ViTPose를 선택하지 않은 이유
ViTPose는 분명히 더 정확하지만:
1. Top-down 구조 → 별도 person detector 필요 → 두 모델 관리 복잡
2. Phase 2 목적은 "논문 최고 성능 재현"이 아니라 "OKS 평가 파이프라인 구축"
3. CARLA(Phase 4) 통합 시 단일 YOLOv8-pose가 더 깔끔한 파이프라인

**향후**: COCO Keypoints 벤치마크에서 SOTA 비교가 필요하면 ViTPose 추가 실험.

### 평가 지표: OKS (Object Keypoint Similarity)

```
OKS = Σ exp(-d_i² / (2 * s² * k_i²)) * δ(v_i > 0) / Σ δ(v_i > 0)
```

| 변수 | 의미 |
|---|---|
| d_i | i번째 keypoint의 예측-GT 거리 |
| s | 객체 크기 (bbox area의 sqrt) |
| k_i | keypoint별 falloff 상수 (COCO 정의) |
| v_i | visibility flag |

OKS를 선택한 이유:
- COCO Keypoint Challenge 공식 지표 → 논문과 직접 비교 가능
- 단순 거리(pixel error)보다 **객체 크기 정규화** — 멀리 있는 사람과 가까운 사람을 공평하게 평가
- k_i(keypoint별 가중치) 덕분에 눈/코처럼 작은 keypoint와 어깨/엉덩이를 구분해서 평가

### 자율주행 연계: 보행자 의도 예측
Pose를 왜 자율주행에서 쓰는가:
- **팔 방향** → 횡단 의도 신호
- **몸통 방향** → 이동 방향 예측
- **걸음 속도 (연속 프레임)** → 급격한 방향 전환 감지

Phase 2에서 skeleton 시각화 + 간단한 rule-based 의도 예측을 구현한 이유:
Phase 4 CARLA 보행자 씬에서 이 로직을 실제로 적용하기 위한 선행 작업.
