# Phase 3 — 설계 근거 문서

> BEV(Bird's Eye View)를 왜 직접 구현했는가, nuScenes 대신 무엇을 썼는가,
> IPM의 한계는 무엇이고 BEVFusion은 어떻게 이를 극복하는가.

---

## 1. BEV — IPM (Inverse Perspective Mapping)

### 목표 설정
BEVFusion, BEVDet 같은 최신 논문을 읽기 전에 **BEV의 기초가 되는 카메라 기하학**을 직접 구현.
핀홀 카메라 모델 → 호모그래피 변환 → 새 관점 영상의 전체 파이프라인을 수식으로 이해.

### 접근 방식 선정: IPM 직접 구현 vs 라이브러리 사용

| 접근 | 특징 | 선택 이유 |
|---|---|---|
| **IPM 직접 구현** | 핀홀 모델 + 호모그래피 수식 직접 작성 | ✅ 원리 이해가 목적 |
| cv2.warpPerspective만 사용 | 한 줄 코드 | ❌ 왜 되는지 모름 |
| nuScenes + BEVDet 코드 가져오기 | 최신 결과 | ❌ Phase 3 단계에서 너무 복잡 |

**결론**: BEVFusion 논문을 읽고 이해하려면 카메라 내부 행렬(K), 외부 행렬(E), 호모그래피(H)의
관계를 몸으로 알아야 함. 라이브러리 의존 없이 직접 구현.

### 핀홀 카메라 모델 설계

```python
# 내부 행렬 K (intrinsic)
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
])
# fx, fy: 초점거리(pixel)
# cx, cy: 주점(이미지 중심)

# 외부 행렬 E (extrinsic) — 카메라 위치/자세
# 카메라가 지면보다 pitch 각도 아래를 향함
```

**설계 결정**: CARLA 카메라 파라미터를 Phase 4와 동일하게 맞춤.
Phase 4에서 `carla_utils.py`의 `get_camera_intrinsic()`과 동일한 K 행렬 사용.

### IPM의 핵심 가정과 한계

**가정**: 지면이 평평하다 (flat ground assumption)

```
영상 좌표 (u,v) → 3D 지면 좌표 → BEV 픽셀 좌표
```

**한계**:
| 상황 | IPM의 문제 |
|---|---|
| 경사로, 언덕 | 지면 평면 가정 위반 → 왜곡 |
| 높이가 있는 물체(차, 사람) | 지면 위 점이 아니므로 BEV 위치 오류 |
| 카메라 흔들림 | 실시간 외부 행렬 갱신 필요 |

이 한계를 직접 확인한 것이 Phase 3의 핵심 학습.
**"왜 BEVFusion이 필요한가"**의 답이 여기 있음.

### BEVFusion이 IPM 한계를 어떻게 극복하는가

| 방법 | 접근 | IPM 대비 장점 |
|---|---|---|
| IPM | 기하학적 변환 (수식) | 빠르지만 평면 가정 필요 |
| LSS (Lift-Splat-Shoot) | 깊이 예측 → 3D voxel | 높이 있는 물체 처리 가능 |
| BEVFusion | Camera + LiDAR fusion | 카메라 한계를 LiDAR로 보완 |
| BEVDet | LSS 기반 카메라 전용 | LiDAR 없이도 합리적 BEV |

**학습 경로**: IPM → LSS 이해 → BEVFusion/BEVDet 논문 읽기.
Phase 3에서 IPM을 직접 구현한 이유가 이 경로를 따르기 위해서.

### 멀티뷰 BEV 설계

nuScenes 6카메라 배치를 모방해 개념 구현:
- Front, Front-Left, Front-Right, Back, Back-Left, Back-Right
- 각 카메라의 K, E 행렬 정의
- 각 뷰의 IPM 결과를 하나의 BEV 캔버스에 합성

**설계 결정**: 실제 nuScenes 데이터 대신 합성 씬 사용.
이유: Phase 3의 목적은 멀티뷰 fusion 원리 이해이지, 대규모 데이터셋 처리가 아님.
nuScenes 실험은 Phase 3 심화 또는 Phase 4 이후 과제.

### Phase 4 연계
CARLA는 멀티카메라 설정 가능. Phase 3에서 구현한 호모그래피 파이프라인을
Phase 4 CARLA 데이터에 그대로 적용하면 실 렌더링 데이터로 BEV 생성 가능.

---

## 2. 핵심 논문 연결

### BEVFusion (2022, MIT)
- **핵심**: Camera feature + LiDAR point cloud를 BEV 공간에서 fusion
- **Phase 3 연결**: IPM이 실패하는 케이스(높이 있는 물체)를 LiDAR depth로 해결
- **읽어야 할 이유**: CARLA에 LiDAR 센서 추가(Phase 4) 시 바로 적용 가능

### BEVDet (2022)
- **핵심**: 카메라만으로 BEV Detection — Lift-Splat-Shoot(LSS) 기반
- **Phase 3 연결**: LSS = "각 픽셀마다 깊이 분포 예측 → 3D 공간에 feature 뿌리기"
- **읽어야 할 이유**: DepthAnythingV2(Phase 1)의 깊이 출력을 LSS 입력으로 쓸 수 있음

### 학습 경로 정리
```
Phase 1 Depth(DepthAnythingV2) → 단안 깊이 이해
Phase 3 IPM → 카메라 기하학 이해
    ↓ 합치면
LSS: 픽셀 깊이 분포 → 3D voxel → BEV feature
    ↓ + LiDAR
BEVFusion: 멀티센서 fusion BEV
```
각 Phase가 독립적이 아니라 누적되는 구조.
