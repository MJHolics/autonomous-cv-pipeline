# Phase 4 — 문제 해결 로그

누적 기록: 발생한 문제, 원인, 해결책을 쌓아가는 파일.

---

## [2026-03-28] CARLA Python API — Python 3.13 미지원

**증상**
Anaconda base (Python 3.13.5)에서 `pip install carla==0.9.15` 실패.
carla wheel이 Python 3.13용 없음.

**해결**
Python 3.10 전용 conda 환경 생성:
```bash
conda create -n carla_env python=3.10 -y
conda activate carla_env
pip install carla==0.9.15
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install numpy opencv-python matplotlib pillow ultralytics transformers timm filterpy lapx supervision
python -m ipykernel install --user --name carla_env --display-name "Python (carla_env)"
```

**교훈**
CARLA 0.9.15 Python API는 cp310(Python 3.10) 전용 wheel만 제공.
Phase 4 관련 노트북은 반드시 `carla_env` 커널 사용.

---

## [2026-03-28] CARLA 서버 실행 방법

**설치 경로:** `C:\Users\apple\Desktop\MISSION\CARLA_0.9.15\WindowsNoEditor\`

**번들 wheel 참고:** `PythonAPI/carla/dist/` 에는 cp37 전용 wheel만 있음.
pip으로 설치한 `carla==0.9.15` (cp310) 사용 — 동일 버전이라 호환.

**Step 1 — CARLA 서버 시작 (별도 터미널):**
```
C:\Users\apple\Desktop\MISSION\CARLA_0.9.15\WindowsNoEditor\CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
```
- 서버 준비 완료 메시지: `listening to client` 로그 확인
- 포트: 2000 (기본), Traffic Manager: 8000

**Step 2 — 데이터 수집 (다른 터미널):**
```bash
conda activate carla_env
cd C:/Users/apple/Desktop/autonomous_cv_pipeline/phase4_carla/data_collection
python carla_data_collector.py --frames 500 --town Town01 --n_npcs 20
```

**저장 경로:** `data_collection/carla_dataset/`

---

## [2026-03-28] 환경 구성 최종 확인

```
carla_env (Python 3.10.20):
  PyTorch     : 2.6.0+cu124  ✅
  CUDA        : True (RTX 4080 SUPER)  ✅
  CARLA API   : 0.9.15  ✅
  OpenCV      : 4.13.0  ✅
  Ultralytics : 8.4.30  ✅
  Transformers: 5.4.0   ✅
```

---

# 실행 결과 해석 로그

---

## [2026-03-29] Phase 4 통합 평가 실행 — 결과 및 원인 분석

### Detection (Precision: 0.1328)

**결과**
```
클래스    TP   FP   FN   Precision  Recall   F1
car       17   15   15   0.5312     0.5312   0.5312
airplane   0    1    0   0.0000     …
bus        0    2    0   0.0000     …
train      0    1    0   0.0000     …
평균                       0.1328     0.1328
```

**원인**
1. **도메인 갭**: YOLOv8을 COCO128(실사 80클래스, 10에폭)로 파인튜닝했지만
   CARLA는 게임엔진 렌더링 → 텍스처·조명이 실사와 다름
2. **클래스 불일치**: CARLA에는 NPC차량이 대부분이나 모델이 airplane/bus/train 등
   드문 클래스에 FP를 내고 있음
3. **학습 부족**: 10에폭은 COCO128 수준에서 적합. CARLA 씬은 별도 파인튜닝 필요

**교훈 / 개선 방향**
- CARLA GT bbox로 YOLOv8 재파인튜닝하면 car 클래스 Precision 0.8+ 예상
- 또는 `yolov8m.pt`(medium) 이상 사용
- 평가 지표는 car 단독 F1=0.53으로 해석하는 게 현실적

---

### Depth (RMSE: 4.83m)

**결과**
```
지표           Phase 1 (합성GT)   Phase 4 (CARLA GT 절대미터)
RMSE(m)↓      0.1600             4.8348   ← 수치 급등
AbsRel↓        0.4318             0.3793   ← 약간 개선
δ1(acc)↑       0.4113             0.4219   ← 약간 개선
```

**원인**
- Phase 1 RMSE 0.16m은 합성 GT(상대 깊이 → 미터 근사)라 실제 의미 없음
- DepthAnythingV2는 **상대 깊이(0~1 정규화)** 출력 → 절대 미터로 변환 시 스케일 정렬 필요
- 스케일 정렬 없이 단순 비교하면 RMSE가 수 미터 단위로 튀는 것은 정상

**교훈 / 개선 방향**
- **Scale-Invariant RMSE(SI-RMSE)** 가 절대값보다 의미있는 비교 지표
- CARLA depth GT로 least-squares 스케일 정렬(s = median(GT/pred)) 후 RMSE 재계산하면
  실질 성능을 정확히 파악 가능
- AbsRel 0.38, δ1 0.42는 Phase 1과 거의 동등 → 모델 자체 품질은 정상

---

### Segmentation (mIoU: 0.0000)

**증상**
SAM2 vs CARLA Semantic GT 비교 차트 미표시, mIoU = 0.0000

**원인 (추정)**
1. **CARLA semantic class ID 불일치**: 코드에서 `person=4, car=10` 가정했으나
   CARLA 버전/맵에 따라 class ID가 다를 수 있음
   - CARLA 0.9.15 기본: pedestrian=4, vehicles=10 (맞음)
   - 단, semantic 카메라 출력이 RGB 인코딩일 경우 R채널 값이 class ID
   - `cv2.IMREAD_GRAYSCALE`로 읽으면 RGB→Gray 변환돼 class ID 손실 가능
2. **GT 마스크 픽셀 부족**: `roi_gt.sum() < 50` 조건으로 대부분 스킵됐을 가능성
3. **SAM2 마스크 shape 불일치**: pred_mask가 (H,W)가 아닌 (1,H,W)일 경우 IoU=0

**해결 방향**
```python
# semantic 이미지를 BGR로 읽고 R채널만 추출
sem_bgr = cv2.imread(str(sem_path))
sem_class = sem_bgr[:, :, 2]  # R채널 = class ID (CARLA 관례)

# SAM2 마스크 차원 확인
pred_mask = res[0].masks.data.cpu().numpy()
if pred_mask.ndim == 3:
    pred_mask = pred_mask[0]  # (1,H,W) → (H,W)
```

**교훈**
CARLA semantic 카메라는 RGB 각 채널에 [R=class, G=tag, B=0] 인코딩.
GRAYSCALE로 읽으면 weighted 변환돼 class ID가 깨짐. 반드시 R채널 단독 추출.

---

### Segmentation 2차 디버깅 — mIoU=0.0000 원인 재분석 (2026-03-29)

**증상 (수정 후에도 동일)**
R채널 추출로 수정했지만 여전히 mIoU=0.0000.

**진단 추가 후 발견**
```
[진단]
  총 이미지: 30 / 라벨 없음: 0 / SAM 마스크 없음: 0
  GT픽셀 부족 스킵: 30 / 실제 평가된 객체: 0

  [000436] semantic 고유 class ID: [ 1  2  5  9 11 15 22 24]
  person(4) 픽셀 수: 0, car(10) 픽셀 수: 0
```

**실제 원인**
`semantic_image_to_mask()`가 2D (H,W) 배열을 `cv2.imwrite`로 저장 → **grayscale PNG**.
→ R/G/B 채널 구분 문제가 아니었음 (어느 채널이든 동일한 값).
→ 진짜 문제: **수집된 데이터에 class 4(Pedestrian), 10(Vehicles) 픽셀이 0**.

**원인 분석**
CARLA 0.9.15에서 NPC 블루프린트에 따라 semantic tag가 다름:
- 일반 vehicle blueprint → class 10 (Vehicles)
- 일부 blueprint 또는 Traffic Manager 소환 차량 → class 20 (Dynamic)
- pedestrian blueprint → class 4 (Pedestrian) 또는 20 (Dynamic)

실제 데이터의 class ID: [1,2,5,9,11,15,20,22,23,24] — class 20(Dynamic)이 차량/보행자일 가능성.

**해결**
GT 마스크 생성 시 class 10과 20을 모두 포함:
```python
VEHICLE_CLASSES    = [10, 20]  # Vehicles + Dynamic
PEDESTRIAN_CLASSES = [4,  20]  # Pedestrian + Dynamic

gt_mask = np.zeros(sem_gt.shape, dtype=np.uint8)
for c in carla_classes:
    gt_mask |= (sem_gt == c).astype(np.uint8)
```

또한 bbox 내부 실제 class ID 진단 코드 추가 → 정확한 매핑 확인 가능.

**3차 버그: `load_gt_boxes(str(lbl_path), ...)` — str vs Path**
```
AttributeError: 'str' object has no attribute 'exists'
```
`load_gt_boxes`가 Path 객체를 기대하는데 `str()`로 감싸서 발생.
→ `str(lbl_path)` → `lbl_path` 로 수정.

**교훈**
1. semantic PNG 저장 방식(2D grayscale)과 읽기 방식을 혼동하지 말 것
2. CARLA NPC semantic 태그는 블루프린트마다 다름 → 수집 전 실제 ID 확인
3. 외부 함수에 인자 타입을 넘길 때 함수 시그니처 먼저 확인 (Path vs str)

---

### Segmentation 5차 — VEHICLE_CLASSES에 class 28 누락 (2026-03-29)

**증상**
2차 수집(보행자 추가) 후에도 car mIoU=0.007로 매우 낮음.

**진단 (`check_dataset.py` 실행)**
```python
# 라벨 있는 프레임의 semantic ID
000190.txt: yolo_classes=[2]  semantic_ids=[ 1  2  5  6  8  9 11 14 22 24 28]
# 전체 클래스 분포
{2: 194}  # car만, person 없음
```

**원인**
CARLA 0.9.15에서 차량 semantic tag가 **class 28** (Car specific).
기존 `VEHICLE_CLASSES = [10, 20]`에서 28이 빠져 GT 마스크 픽셀이 0 → mIoU=0.007.

```
CARLA_SEMANTIC (0.9.15 실제):
  10: Vehicles (general)
  20: Dynamic
  28: Car (specific, 0.9.15에서 추가)
```

**해결**
```python
VEHICLE_CLASSES = [10, 20, 28]  # class 28 추가
```

**교훈**
carla_utils.py의 CARLA_SEMANTIC 딕셔너리가 0~22만 정의되어 있어 28이 누락됨.
외부 시스템(CARLA) 버전이 올라가면 semantic 태그도 바뀔 수 있음.
→ 수집 후 반드시 `check_dataset.py` 로 실제 class ID 확인 필수.

---

### Segmentation 4차 — person 클래스 데이터 자체가 없음 (2026-03-29)

**증상**
seg_results에 'car'만 있고 'person'이 없음. car mIoU=0.030 (비정상적으로 낮음).

**원인 1 — 보행자 미스폰**
`carla_data_collector.py`에 NPC 차량 스폰 코드만 있고 **보행자(walker) 스폰 코드 없음**.
`get_actor_bboxes_2d()`는 `walker.pedestrian.*`을 필터링하도록 구현돼 있었지만
월드에 보행자 자체가 없어서 label에 person이 한 번도 안 나옴.

**해결**
`carla_data_collector.py`에 보행자 스폰 로직 추가:
```python
walker_bps = bp_lib.filter('walker.pedestrian.*')
walker_ctrl_bp = bp_lib.find('controller.ai.walker')
# 내비게이션 메시 기반 랜덤 위치에 스폰
# controller.ai.walker로 자율 보행 활성화
```
`--n_pedestrians` 인자 추가 (기본값 15명).

**원인 2 — SAM2 마스크 해상도 불일치**
car mIoU=0.030이 나온 이유: SAM2가 내부 640×640 해상도로 마스크 출력.
sem_gt는 CARLA 원본 1280×720 → 두 마스크 shape 불일치 → IoU 거의 0.
→ pred_mask를 sem_gt 해상도로 `cv2.resize` 후 IoU 계산하도록 수정.

**재수집 명령어**
```bash
conda activate carla_env
cd phase4_carla/data_collection
python carla_data_collector.py --frames 500 --n_npcs 20 --n_pedestrians 15
```

**보행자 라벨 미생성 추가 분석**
재수집 후에도 label class 분포 = `{2: 194}` — person(0) 없음.
`get_actor_bboxes_2d`에서 `walker.pedestrian.*`을 필터링하지만
→ bbox 8꼭짓점 중 카메라 앞(z>0)에 4개 미만이면 스킵
→ 보행자가 카메라 시야 각도 밖 또는 Town01 도로변 내비게이션 메시 위치와 에고 차량 경로가 겹치지 않음
→ semantic에는 class 4가 찍히지만 bbox projection 실패로 라벨 생성 안 됨.

**개선 방향**
1. `--n_pedestrians` 수 대폭 증가 (50+)
2. `max_distance` 완화 (50→80m) + `< 4 points` → `< 2 points`로 완화
3. 또는 ego 차량 스폰 포인트 근처에 pedestrian 강제 스폰

**교훈**
"car만 나오는 게 정상"이라는 성급한 결론을 내리지 말 것.
스크립트 코드 + 실제 데이터를 직접 확인(check_dataset.py)해서 검증 후 판단해야 함.
평가 데이터의 다양성(클래스 분포)은 수집 스크립트 설계 단계에서 보장해야 함.

---

### Segmentation 6차 — 3차 수집 후 person만 나오고 car 없음 (2026-03-29)

**3차 수집 결과 (--n_npcs 30 --n_pedestrians 30)**
```
[bbox 내부 진단] 첫 5장: YOLO=person만 등장 (car 없음)

[Segmentation mIoU]
  person: mIoU=0.2495 (n=13)  ← 드디어 person 등장!
  car: 없음

[Tracking]
  MOTA=-0.0458, TP=23, FP=30, FN=130, ID_SW=0
```

**진전**
- person semantic class 4가 bbox 안에 확인됨 → `PEDESTRIAN_CLASSES=[4,20]` 정상 동작
- SAM2 마스크 리사이즈 적용 후 mIoU=0.2495 (0.007에서 대폭 향상)

**car가 없는 이유 (신규 분석)**
`carla_data_collector.py`에 `get_vehicle_class()` 도입 + `save_yolo_annotation` class_map 변경 후
라벨 class 분포가 달라졌을 가능성.
eval 노트북의 `cls_name = 'person' if cls_id == 0 else 'car'` 로직이
새 class_map(bicycle=1, car=2, motorcycle=3, bus=5, truck=7)을 처리 못함.

**내일 해결 목록**
1. `check_dataset.py` 재실행 → 3차 수집 데이터 class 분포 확인
2. evaluation 노트북 `cls_name` 로직을 다중 클래스 처리로 업데이트
   - 현재: `'person' if cls_id == 0 else 'car'` (2개뿐)
   - 필요: `{0:'person', 1:'bicycle', 2:'car', 3:'motorcycle', 5:'bus', 7:'truck'}.get(cls_id, 'unknown')`
3. `VEHICLE_CLASSES` 에 실측 확인된 semantic ID 추가 (현재 [10,20,28])
4. 데이터 재수집 필요 여부 판단

---

### MOTA 음수 — 그래프 표기 문제 (메모)

**증상**
MOTA=-0.0458로 음수 → 대시보드 bar chart가 음수를 처리 못해 그래프 이상하게 표기됨.

**원인**
`axes[1][0].bar(...)` + `ax.set_ylim(0, 1.1)` → 음수 막대가 y=0 아래로 내려가지만
ylim이 0 고정이라 잘려서 안 보임.

**내일 수정 방향 (두 가지 옵션)**

옵션 A — ylim 동적 조정 (빠른 수정):
```python
mota_min = min(0, trk_mota) - 0.05
ax.set_ylim(mota_min, 1.1)
ax.axhline(y=0, color='black', linewidth=0.8)  # 0 기준선 표시
```

옵션 B — MOTA 해석 텍스트로 보완 (더 나은 방법):
음수 MOTA는 Detection 품질 문제임을 텍스트로 명시하고
별도 subplot에 TP/FP/FN 비율 막대 차트로 원인 시각화.
```python
# TP/FP/FN breakdown 차트 추가
labels = ['TP', 'FP', 'FN']
values = [tracking_result['tp'], tracking_result['fp'], tracking_result['fn']]
colors = ['#27AE60', '#E74C3C', '#E74C3C']
ax.bar(labels, values, color=colors, alpha=0.85)
ax.set_title(f'Tracking 상세\nMOTA={trk_mota:.4f}')
```

**근본 해결**: Detection 성능 개선 (CARLA 데이터로 YOLOv8 파인튜닝) → MOTA 양수 회복

---

---

### Segmentation 7차 — car bbox와 semantic 위치 불일치 (2026-03-30)

**진단 결과**
```
000085 | YOLO=car | bbox IDs: [1 2 5 6 9 14] | class4=0 class28=0 class20=0
000086 | YOLO=car | bbox IDs: [1 2 5 6 9 14] | class4=0 class28=0 class20=0
```
car bbox 내부에 Building(1)/Fence(2)/Pole(5)/RoadLine(6)/Vegetation(9)/Ground(14)만 존재.
class 28(Car) 픽셀이 0 → bbox와 semantic 위치가 완전히 엇나감.

**비교: person은 정상**
```
000000 | YOLO=person | bbox IDs: [2 4 22] | class4=219 class28=0 class20=0
```
person bbox에 class 4(Pedestrian) 219픽셀 정상 포함.

**원인 분석**
`get_actor_bboxes_2d()`에서 차량 3D bbox를 2D로 투영할 때
CARLA 좌표계 변환 오류:
```python
# 현재 코드
pt_cam_cv = np.array([pt_cam[1], -pt_cam[2], pt_cam[0]])
```
- CARLA World: X=forward, Y=right, Z=up (UE4 좌표계)
- CARLA Camera: X=forward, Y=right, Z=down
- OpenCV Camera: X=right, Y=down, Z=forward

차량 bounding box의 8꼭짓점 중 일부가 카메라 뒤로 넘어가면
투영 결과가 화면 밖으로 벗어나 semantic과 매칭이 안 됨.
person은 사람 크기가 작아서 8꼭짓점이 대부분 카메라 앞에 있어 잘 맞음.

**결과**
car mIoU=0.006은 SAM2 품질이 아닌 **bbox 투영 정확도 문제**.
현재 구현으로는 car semantic 평가 불가. person 기준 mIoU=0.147이 실질 지표.

**개선 방향 (다음 개발 시)**
1. `get_actor_bboxes_2d`에서 투영된 bbox와 semantic mask 겹침 비율 사전 검증
2. 또는 semantic mask에서 직접 bbox 추출 (top-down 방식으로 역산)
3. CARLA `carla.Client.apply_batch`로 GT를 직접 받는 방식도 검토

---

### Phase 4 최종 결과 (2026-03-30)

```
[Detection]   person F1=0.419, car F1=0.214  ← COCO 도메인 갭
[Depth]       RMSE=5.09m, AbsRel=0.38, δ1=0.42  ← 스케일 미정렬, 구조 파악 정상
[Segmentation] person mIoU=0.147, car mIoU=0.006(bbox 투영 오류)
[Tracking]    MOTA=-0.046  ← Detection FN=130이 병목, ByteTrack 자체는 정상

대시보드 저장: phase4_carla/evaluation/final_dashboard.png
```

---

## 현재 수정된 파일 목록 (2026-03-30 기준)

| 파일 | 변경 내용 |
|---|---|
| `carla_utils.py` | CARLA_SEMANTIC에 class 28 추가, BLUEPRINT_TO_CLASS + get_vehicle_class() 추가, save_yolo_annotation class_map 확장(bicycle/motorcycle/bus/truck) |
| `carla_data_collector.py` | 보행자 스폰 추가, 4륜/2륜 혼합 NPC 스폰, --n_pedestrians 인자 추가 |
| `evaluation/01_integrated_evaluation.ipynb` | VEHICLE_CLASSES=[10,20,28], SAM2 마스크 리사이즈, f-string 버그 수정, 진단 코드 추가 |

---

### Tracking (MOTA: 0.2500)

**결과**
```
Phase 2 (시뮬): MOTA=0.9412, ID_SW=0
Phase 4 (CARLA): MOTA=0.2500, TP=16, FP=8, FN=16, ID_SW=0
```

**원인**
- Detection 품질 저하(Precision 0.53)가 트래킹 FN 직접 유발
- MOTA = 1 - (FP+FN+ID_SW)/GT_total
- FN=16, FP=8 → GT 오브젝트가 ~32개일 때 MOTA=(32-24)/32=0.25 공식 일치
- ID_SW=0은 유지 → ByteTrack 매칭 로직 자체는 정상

**교훈**
Tracking 성능의 병목은 Detector 품질. Detection 개선 시 MOTA 자동 상승.
Phase 4 개선 우선순위: Detection 재파인튜닝 → Tracking 재평가 순서.

---

## [2026-03-29] .venv 환경으로 실행 시도 오류

**증상**
```
python carla_data_collector.py --frames 500
can't open file: [Errno 2] No such file or directory
```

**원인**
1. `.venv` 활성화 상태 (CPU-only, carla 미설치)
2. 실행 경로가 프로젝트 루트 → 파일은 `phase4_carla/data_collection/` 안에 위치

**해결**
```powershell
deactivate
conda activate carla_env
python phase4_carla/data_collection/carla_data_collector.py --frames 500 --town Town01 --n_npcs 20
```

**교훈**
Phase 4 스크립트는 항상 `carla_env` + 절대경로 또는 `phase4_carla/data_collection/` 기준 실행.

---

## [2026-03-30] `conda run -n autonomous_cv` 실패 — Jupyter 커널 ≠ conda 환경

**증상**
Claude가 노트북 작성 전 사전 분석을 위해 외부 Python 스크립트를 실행하려 했음:
```bash
conda run -n autonomous_cv python -c "import cv2, numpy as np ..."
# → EnvironmentLocationNotFound: Not a conda environment
```

이어서 `carla_env`로 시도:
```bash
conda run -n carla_env python -c "
from pathlib import Path
import numpy as np ...
"
# → AssertionError: Support for scripts where arguments contain newlines not implemented.
```

**원인**
두 가지 별개의 문제:

1. **`autonomous_cv`는 conda 환경이 아님**
   - `conda env list`에 없음. Jupyter 커널 이름(`--name autonomous_cv`)으로만 등록된 것.
   - 실제 환경은 Anaconda base(`C:\Users\apple\Anaconda3`) 또는 다른 경로.
   - Jupyter 커널 이름과 conda 환경 이름은 별개 개념 — 커널이 어떤 Python을 쓰는지는
     `jupyter kernelspec list`로 확인해야 함.

2. **`conda run -c "멀티라인"`은 Windows conda에서 미지원**
   - `conda run`의 `-c` 인자에 개행(`\n`) 포함 시 AssertionError 발생.
   - Windows conda 25.7.0 제약.

**해결 (채택한 방법)**
사전 분석 코드를 노트북 Cell 안에 넣어 실행.
- 노트북은 이미 올바른 커널(`autonomous_cv`)에서 실행 중 → 환경 문제 없음.
- 별도 `.py` 파일 작성 후 경로 지정 실행도 가능하지만 불필요한 파일 생성.

**대안 (다음에 외부 스크립트가 필요한 경우)**
```bash
# 방법 1: 스크립트 파일로 분리
python "C:/Users/apple/Anaconda3/python.exe" script.py

# 방법 2: conda run에 파일 경로 사용 (개행 없음)
conda run -n carla_env python path/to/script.py

# 방법 3: Anaconda base Python 직접 사용
"C:/Users/apple/Anaconda3/python.exe" -c "단일라인코드"
```

**교훈**
- Jupyter 커널 이름 ≠ conda 환경 이름. `jupyter kernelspec list`로 커널→Python 경로 확인.
- 멀티라인 분석 코드는 노트북 Cell이나 `.py` 파일로 분리하는 것이 안전.
- Claude가 노트북 외부에서 Python을 실행할 때는 `conda run`보다
  직접 Python 경로 지정이 더 신뢰성 있음.
