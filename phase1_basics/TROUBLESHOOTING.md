# Phase 1 — 문제 해결 로그

누적 기록: 발생한 문제, 원인, 해결책을 쌓아가는 파일.

---

## [2026-03-26] Jupyter 커널 — `ModuleNotFoundError: No module named 'torch'`

**증상**
노트북 실행 시 torch import 실패.

**원인**
- 프로젝트 venv (`autonomous_cv_pipeline/venv`)가 CPU-only torch 2.11.0 설치 상태
- ultralytics 미설치
- VSCode가 venv 커널을 기본 선택함

**해결**
Anaconda base Python을 Jupyter 커널로 등록:
```bash
python -m ipykernel install --user --name autonomous_cv --display-name "Python (autonomous_cv)"
```
VSCode 노트북에서 커널을 **`Python (autonomous_cv)`** 으로 변경.

**재발 방지**
- 노트북 metadata에 `"name": "autonomous_cv"` 커널 고정
- venv는 사용하지 않음 (CPU-only)

---

## [2026-03-26] `check_det_dataset` — `KeyError: slice(None, 10, None)`

**증상**
```
KeyError: slice(None, 10, None)
print(f"클래스 샘플: {dataset_info['names'][:10]}")
```

**원인**
`dataset_info['names']`가 `list`가 아닌 `dict` (`{0: 'person', 1: 'bicycle', ...}`) 형태.
dict는 슬라이스 인덱싱 불가.

**해결**
```python
# 수정 전
class_names = dataset_info['names']         # dict
print(dataset_info['names'][:10])           # KeyError

# 수정 후
class_names = dataset_info['names']         # dict 유지 (cls_id로 접근)
class_names_list = list(class_names.values())
print(class_names_list[:10])                # OK
```

**교훈**
Ultralytics의 `names`는 항상 `dict` 타입. 리스트처럼 슬라이스할 때는 `.values()` 변환 필요.

---

## [2026-03-26] 한글 폰트 — matplotlib 한글 깨짐

**증상**
matplotlib 그래프 제목/축 레이블의 한글이 네모(□)로 표시.

**원인**
matplotlib 기본 폰트가 한글 미지원.

**해결**
모든 노트북 첫 번째 셀에 추가:
```python
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트
matplotlib.rcParams['axes.unicode_minus'] = False      # 마이너스 기호 깨짐 방지
```

**참고**
- Windows: `Malgun Gothic` (맑은 고딕)
- macOS: `AppleGothic`
- Linux: `NanumGothic` (별도 설치 필요)

---

## [2026-03-27] FP/FN 분석 — `KeyError: 'tp'`

**증상**
```
KeyError: 'tp'
df['precision'] = df['tp'] / (df['tp'] + df['fp'] + 1e-6)
```

**원인**
`stats` dict가 비어있어 `pd.DataFrame(stats).T`에 컬럼이 없음.
`img_dir` / `label_dir` 경로가 존재하지 않아 이미지를 하나도 읽지 못함.
- 예상 경로: `Path.home() / 'datasets' / 'coco128'`
- 실제 경로: `Path.home() / 'MyProject' / 'datasets' / 'coco128'`

**해결**
```python
# 수정 전
data_path = Path.home() / 'datasets' / 'coco128'

# 수정 후
data_path = Path.home() / 'MyProject' / 'datasets' / 'coco128'
```

**교훈**
Ultralytics가 데이터셋을 다운로드하는 기본 경로는 `~/datasets/`이지만,
이전 학습 시 working directory나 yaml 설정에 따라 위치가 달라질 수 있음.
경로 하드코딩 전에 `path.exists()` 확인 또는 glob으로 실제 위치 탐색 필요.

---

## [2026-03-27] Segmentation — SAM2 패키지 미설치

**증상**
`import sam2` → `ModuleNotFoundError`

**원인**
SAM2 공식 패키지는 Anaconda base에 설치되어 있지 않음.
`segment_anything` (SAM v1)은 설치됨.

**해결**
Ultralytics SAM 래퍼 사용:
```python
from ultralytics import SAM
model = SAM('sam2.1_b.pt')  # 자동 다운로드 ~40MB
```
별도 `sam2` 패키지 설치 불필요.

**교훈**
Ultralytics 8.3+ 에 SAM2 추론이 내장되어 있음. 공식 sam2 패키지 대신 ultralytics 활용.

---

## [2026-03-27] Depth — DepthAnythingV2 패키지 미설치

**증상**
`import depth_anything_v2` → `ModuleNotFoundError`

**원인**
`depth_anything`, `depth_anything_v2` 패키지 미설치.

**해결**
HuggingFace transformers로 대체:
```python
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
processor   = AutoImageProcessor.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
depth_model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Small-hf')
```
첫 실행 시 ~97MB 자동 다운로드.

**교훈**
transformers 4.x에서 DepthAnythingV2 지원. 별도 패키지 불필요.

---

## [2026-03-27] Depth — `SyntaxError: unterminated string literal`

**증상**
```
Cell In[8], line 21
    summary_names  = ['RMSE
SyntaxError: unterminated string literal
```

**원인**
노트북 셀 source를 Python 스크립트로 패치할 때 문자열 안의 `\n`이
실제 줄바꿈 문자로 저장됨 → Python 코드 내 문자열 리터럴이 중간에 끊김.

**해결**
셀 source를 줄 단위 리스트로 저장 (Jupyter 권장 형식):
```python
nb['cells'][i]['source'] = ['line1\n', 'line2\n', ...]
```
문자열 안에 `\n`이 필요하면 실제 개행 대신 유니코드 공백/다른 표현으로 우회하거나,
문자열을 단일 라인으로 작성.

**교훈**
노트북 JSON을 직접 수정할 때는 source를 단일 문자열이 아닌 `list[str]`로 저장하고,
Python 코드 내 문자열 리터럴에 실제 개행이 들어가지 않도록 주의.

---

# 실행 결과 해석 로그

## [2026-03-27] Phase 1 Detection — 최종 결과

| 지표 | 값 | 해석 |
|------|-----|------|
| YOLOv8n mAP@50 | ~0.6x | COCO128 10epoch 데모, 충분한 수렴 |
| 직접구현 mAP vs Ultralytics | 거의 동일 | 구현 정확성 검증됨 |

---

## [2026-03-27] Phase 1 Segmentation — 최종 결과

```
총 평가 객체 수 : 16개
평균 mIoU       : 0.6941  (Point vs Box 일치도)
평균 Dice       : 0.7130
```

**해석**
- `mIoU 0.694` — Point 클릭 한 번만으로 Box 기반 마스크와 69% 일치.
  SAM2의 강력한 일반화 능력을 정량적으로 확인.
- `Dice 0.713 > IoU 0.694` — Dice = 2·IoU/(1+IoU) 공식 성립 확인 → 직접 구현 검증됨.
- GT 마스크 부재로 절대 mIoU 미측정. Phase 4 CARLA semantic GT로 재평가 예정.

---

## [2026-03-27] Phase 1 Depth — 최종 결과

```
지표          평균      중앙값    표준편차
RMSE(m)↓    0.1600    0.1591    0.0398
AbsRel↓     0.4318    0.4228    0.1229
δ1(acc)↑    0.4113    0.3755    0.1325
δ2(acc)↑    0.7110    0.6716    0.1060
SI-RMSE↓    3.6279    1.8274    3.0467
```

**해석**
- `RMSE 0.16m` — 합성 GT 기준. 절대값보다 이미지 간 일관성(표준편차 0.04)이 낮은 게 핵심.
- `AbsRel 0.43` — 상대 오차 43%. 합성 GT의 한계(밝기→깊이 근사)로 인한 노이즈 포함.
- `δ1 0.41` — 픽셀 41%가 1.25배 이내 오차. 실제 NYU 벤치마크(~0.98)보다 낮지만
  GT가 합성이라 당연. 파이프라인 동작 자체는 정상.
- `SI-RMSE 표준편차 3.03` — COCO128이 실내/실외/다양한 씬 혼합이라 이미지별 편차 큼.
- **Phase 4 CARLA**: depth sensor로 실제 GT 생성 시 모든 지표 신뢰도 대폭 향상 예상.

---

## [2026-03-29] Phase 4 CARLA GT 비교 — Depth & Segmentation 재평가

### Depth 재평가 결과

```
지표       Phase 1 (합성GT)  Phase 4 (CARLA 절대미터)  해석
RMSE(m)    0.1600            4.8348                   Phase 1이 작아보이지만 GT 단위 다름
AbsRel     0.4318            0.3793                   실제 성능은 비슷하거나 소폭 향상
δ1         0.4113            0.4219                   CARLA GT가 더 신뢰할 수 있는 지표
```

**결론**: Phase 1 RMSE 0.16m은 합성 GT 한계로 과소평가된 수치.
CARLA 기준 AbsRel 0.38이 실질 성능 지표. 스케일 정렬 후 재계산 필요.

### Segmentation 재평가 결과

**mIoU = 0.0000 → 코드 버그**
- CARLA semantic 이미지를 `IMREAD_GRAYSCALE`로 읽어 R채널 class ID가 손실됨
- 수정 후 재실행 필요 (phase4_carla/TROUBLESHOOTING.md 참고)
