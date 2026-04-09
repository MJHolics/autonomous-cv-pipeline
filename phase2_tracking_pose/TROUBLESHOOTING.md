# Phase 2 — 문제 해결 로그

누적 기록: 발생한 문제, 원인, 해결책을 쌓아가는 파일.

---

## [2026-03-27] YOLO 모델 경로 — `FileNotFoundError`

**증상**
```
FileNotFoundError: No such file or directory: 'runs\detect\coco128_finetune\weights\best.pt'
```

**원인**
Phase 2 노트북의 working directory는 `phase2_tracking_pose/tracking/`.
Phase 1에서 학습된 `best.pt`의 상대경로가 맞지 않음.

**해결**
```python
# 수정 전
model = YOLO('runs/detect/coco128_finetune/weights/best.pt')

# 수정 후
from pathlib import Path
project_root = Path('C:/Users/apple/Desktop/autonomous_cv_pipeline')
model = YOLO(project_root / 'phase1_basics/detection/runs/detect/coco128_finetune/weights/best.pt')
```

**교훈**
다른 Phase 노트북에서 Phase 1 산출물 참조 시 항상 절대경로 사용.

---

## [2026-03-27] `compute_mota` — `TypeError: tuple indices must be integers`

**증상**
```
TypeError: tuple indices must be integers or slices, not str
pred_tracks = pred_frame['tracks']
```

**원인**
`simulate_tracking()`은 `(frame_id, tracks)` 튜플 리스트를 반환.
`compute_mota()`는 `{'tracks': ...}` dict 형태를 기대.
두 함수의 반환 형식 불일치.

**해결**
```python
# MOTA 셀에서 변환 후 전달
sim_results_dicts = [{'frame_id': f, 'tracks': tracks} for f, tracks in sim_results]
mota_result = compute_mota(sim_results_dicts, gt_frames)
```

**교훈**
함수 간 데이터 형식을 통일하거나, 호출 전에 명시적으로 변환.

---

# 실행 결과 해석 로그

## [2026-03-27] Phase 2 Tracking (ByteTrack) — 최종 결과

```
MOTA  : 0.9412
ID_SW : 0      (ID Switch 없음)
```

**해석**
- `MOTA 0.9412` — 시뮬레이션 씬 기준 매우 우수. ByteTrack의 핵심인
  "low-confidence 박스를 버리지 않고 2단계 매칭"이 ID Switch 0을 만든 핵심 원인.
- `ID_SW = 0` — 한 번도 객체 ID가 바뀌지 않음. 실제 드라이빙 영상(빠른 이동, 가림)에서는
  낮아질 수 있음 → Phase 4 CARLA 동적 씬에서 재평가 예정.
- 시뮬레이션 데이터라 실제 벤치마크(MOT17 기준 MOTA ~0.77)와 직접 비교 불가.
  파이프라인 동작 정확성 확인이 목적.

---

## [2026-03-27] Phase 2 Pose Estimation — 최종 결과

- OKS 기반 keypoint 평가 파이프라인 구현 완료
- YOLOv8-pose + Skeleton 시각화 + 보행자 의도 예측 구현
- (실행 결과 수치 수집 필요 — 다음 세션에 추가)

---

## [2026-03-29] Phase 4 CARLA GT 비교 — Tracking 재평가

```
Phase 2 (시뮬): MOTA=0.9412, ID_SW=0
Phase 4 (CARLA): MOTA=0.2500, TP=16, FP=8, FN=16, ID_SW=0
```

**해석**
- MOTA 급락(0.94 → 0.25)의 원인은 ByteTrack이 아니라 **Detection 품질** 문제
- ID_SW=0 유지 → ByteTrack 매칭 알고리즘은 CARLA 실 씬에서도 정상 동작
- FN이 많은 이유: YOLOv8 COCO128 모델이 CARLA 씬에서 차량을 못 잡음
- **Detection 재파인튜닝 → MOTA 재측정** 이 다음 개선 순서
