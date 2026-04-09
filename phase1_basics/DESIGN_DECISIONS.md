# Phase 1 — 설계 근거 문서

> "왜 이 모델을 골랐는가, 무엇을 배웠는가"를 기록하는 파일.
> 코드는 노트북에, 결과는 TROUBLESHOOTING.md에 — 여기에는 **의사결정 근거**만 쌓는다.

---

## 1. Detection — YOLOv8

### 목표 설정
단순히 YOLO API를 호출하는 것이 아니라, **mAP를 직접 구현**해서 "왜 이 숫자가 나오는지"를 이해하는 것.
CARLA 파이프라인(Phase 4)에서 GT bbox가 존재하는 환경을 미리 설계해두는 것도 목표.

### 모델 선정 이유: YOLOv8n

| 후보 | 장점 | 단점 | 선택 여부 |
|---|---|---|---|
| YOLOv8n (nano) | 경량, 빠른 실험, Ultralytics 생태계 | 정확도 낮음 | ✅ 선택 |
| YOLOv8m (medium) | 정확도 높음 | 학습 시간 길고 Phase 1 목적과 맞지 않음 | ❌ |
| RT-DETR | Transformer 기반 최신 SOTA | 구조 복잡, Phase 1 입문에 부적합 | ❌ |
| Faster R-CNN | 2-stage, 원리 학습에 좋음 | 속도 느림, Ultralytics 외 환경 필요 | ❌ |

**결론**: Phase 1의 핵심은 "모델 성능"이 아니라 **평가 파이프라인 직접 구현**. 가장 빠르게 돌아가는 YOLOv8n이 적합.

### mAP 직접 구현 이유
Ultralytics의 `model.val()`은 내부에서 mAP를 계산해주지만 블랙박스.
직접 구현한 이유:
- IoU threshold를 바꿀 때 내부에서 무슨 일이 일어나는지 이해
- TP/FP/FN을 직접 세며 Precision-Recall 곡선의 의미를 체화
- CARLA GT (Phase 4) 포맷에 맞게 커스텀 평가기를 재사용하기 위해

### 핵심 설계 결정: IoU 계산 직접 구현 vs torchvision.ops
`torchvision.ops.box_iou`를 쓰면 한 줄이지만, Phase 4에서 CARLA bbox 포맷 변환이 필요하다.
직접 구현한 코드가 어떤 포맷도 받을 수 있도록 유연성을 확보.

### 학습 포인트
- `dataset_info['names']`는 `dict` → `.values()` 변환 필요 (TROUBLESHOOTING 참고)
- mAP@50은 IoU=0.5 단일 threshold, mAP@50:95는 0.5~0.95 평균 — 두 지표의 차이를 직접 확인
- COCO128 10에폭 결과가 Phase 4 CARLA에서 Precision 0.53으로 떨어진 이유: **도메인 갭**
  → 모델이 본 적 없는 렌더링 스타일(게임엔진)에 대한 일반화 한계

---

## 2. Segmentation — SAM2 (via Ultralytics)

### 목표 설정
SAM을 "쓰는 것"에서 "평가하는 것"으로. 마스크 IoU를 직접 구현하고,
Point prompt vs Box prompt의 성능 차이를 정량적으로 비교.

### 모델 선정 이유: SAM2.1-base

| 후보 | 장점 | 단점 | 선택 여부 |
|---|---|---|---|
| SAM2.1-base (40MB) | 빠른 추론, 충분한 성능 | tiny보다 약간 무거움 | ✅ 선택 |
| SAM2.1-large | 최고 성능 | 400MB+, Phase 1 실험에 오버스펙 | ❌ |
| SAM v1 (segment_anything) | 원조 논문 | SAM2보다 성능 낮음, 비디오 미지원 | ❌ |
| Mask R-CNN | 파인튜닝 가능 | SAM처럼 zero-shot 불가, Phase 1 취지와 맞지 않음 | ❌ |

**결론**: Phase 1은 zero-shot 일반화 성능 측정이 목적. 파인튜닝 없이 쓸 수 있는 SAM2가 가장 적합.

### Prompt 전략: Box > Point 선택 이유
- **Point prompt**: 클릭 위치 하나 → 모델이 대상을 추측 → 모호성 존재
- **Box prompt**: GT bbox 제공 → 대상 영역이 명확 → IoU가 더 높고 재현 가능
- Phase 1 목표: 마스크 품질 자체 평가 → Box prompt가 더 공정한 기준

### 평가 지표 선택: Mask IoU + Dice

| 지표 | 수식 | 선택 이유 |
|---|---|---|
| Mask IoU | \|P∩G\| / \|P∪G\| | segmentation 표준 지표, COCO 벤치마크 기준 |
| Dice | 2\|P∩G\| / (\|P\|+\|G\|) | IoU와 상관관계 검증 — Dice = 2IoU/(1+IoU) 공식 확인 |

둘 다 구현한 이유: **구현 정확성 교차 검증**. Dice가 IoU의 단조 증가함수이므로 둘의 순위가 같아야 함.

### Phase 4 연계: CARLA Semantic GT 활용
Phase 1에서는 GT 마스크가 없어 Point vs Box 비교만 가능했음.
Phase 4 CARLA semantic 카메라가 픽셀 단위 GT를 제공 → 절대 mIoU 측정 가능.
→ 이 설계 결정이 Phase 4 평가 파이프라인을 미리 염두에 두고 만든 이유.

### 핵심 버그 및 교훈 (Phase 4 연계)
CARLA semantic PNG를 `IMREAD_GRAYSCALE`로 읽으면 R채널의 class ID가 손실됨.
→ OpenCV BGR 읽기 후 `[:,:,2]` R채널 추출 필요.
→ 이 버그가 Phase 4 mIoU=0.0000의 원인 (2026-03-29 수정).

---

## 3. Depth — DepthAnythingV2

### 목표 설정
단안 깊이 추정(Monocular Depth Estimation)의 **평가 파이프라인** 구축.
RMSE, AbsRel, δ1 등 표준 지표를 직접 구현하고 Phase 4 절대 GT와 비교 가능한 구조 설계.

### 모델 선정 이유: DepthAnythingV2-Small (HuggingFace)

| 후보 | 장점 | 단점 | 선택 여부 |
|---|---|---|---|
| DepthAnythingV2-Small | 경량(97MB), HuggingFace 지원 | 상대 깊이 출력 | ✅ 선택 |
| DepthAnythingV2-Large | 최고 성능 | 400MB+, 추론 느림 | ❌ |
| MiDaS | 검증된 모델 | DepthAnythingV2 대비 구식 | ❌ |
| ZoeDepth | 절대 깊이 출력 | 설치 복잡, Phase 1 취지에 오버스펙 | 후보 유지 |

**결론**: 논문(2024)이 최신이고, HuggingFace transformers로 설치 없이 바로 사용 가능. Small 모델로 빠른 실험.

### 모델 선택의 아키텍처 근거
DepthAnythingV2의 핵심 기여:
1. **합성 데이터(Synthetic data)로 pretraining** → 고품질 pseudo GT 생성
2. **DINOv2 인코더** → 강력한 visual feature 추출
3. **상대 깊이(relative depth)** 출력 → 스케일 불명확하지만 구조 파악 탁월

Phase 1에서 이 상대 깊이의 한계를 직접 확인하는 것이 학습 목표.

### 평가 지표 선택

| 지표 | 수식 | 의미 |
|---|---|---|
| RMSE | √(Σ(pred-gt)²/n) | 절대 오차, 스케일 영향 받음 |
| AbsRel | Σ\|pred-gt\|/gt / n | 상대 오차, 스케일 중립적 |
| δ1 | % pixels where max(pred/gt, gt/pred) < 1.25 | 정확도 지표 |
| SI-RMSE | Scale-Invariant RMSE | 스케일 정렬 후 구조적 오차 측정 |

**AbsRel과 δ1을 핵심으로 선택한 이유**: 상대 깊이 모델은 RMSE가 스케일에 따라 크게 변함.
Phase 1(합성 GT) → Phase 4(절대 미터 GT) 비교 시 RMSE가 폭등하는 이유가 바로 이것.

### Phase 1 vs Phase 4 비교에서 얻은 인사이트

```
지표       Phase 1 (합성GT)  Phase 4 (CARLA 절대GT)  해석
RMSE       0.16m             4.83m                  스케일 미정렬 → 의미 없는 비교
AbsRel     0.43              0.38                   실제 성능은 비슷 or 소폭 향상
δ1         0.41              0.42                   구조 파악 능력은 일관됨
```

**학습**: RMSE만 보면 Phase 4가 훨씬 나빠보이지만 실제로는 아님.
올바른 비교는 **스케일 정렬(median scaling) 후 RMSE** 또는 **AbsRel/δ1** 기준이어야 함.

### 개선 방향
```python
# 스케일 정렬: least-squares median scaling
scale = np.median(gt_depth) / np.median(pred_depth)
pred_aligned = pred_depth * scale
# 이후 RMSE 재계산 → 실질 성능 확인 가능
```
