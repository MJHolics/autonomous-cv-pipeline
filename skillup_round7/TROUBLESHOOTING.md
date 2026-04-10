# skillup_round7 — 문제 해결 로그

누적 기록: 발생한 문제, 원인, 해결책.

---

## [01 Stereo Depth] disparity=0 픽셀 → depth=inf 문제

**증상**
`disp_to_depth()` 변환 시 disparity=0인 픽셀에서 depth가 inf로 터짐.
EPE 계산에서 NaN/inf 섞여서 평균값이 폭발.

**원인**
`depth = focal × baseline / disparity` — disparity=0이면 0 나눔.
합성 데이터에서 occluded region, padding 영역이 disp=0으로 채워짐.

**해결**
```python
def disp_to_depth(disp, focal=FOCAL_LENGTH, baseline=BASELINE):
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = disp > 0
    depth[valid] = focal * baseline / disp[valid]  # valid 영역만 변환
    return depth
```
평가 함수에서도 `valid = (gt_disp > 0) & (gt_disp < max_disp)` mask 적용.

**교훈**
disparity 기반 depth 변환에서는 항상 valid mask 필수.
LiDAR GT depth에서도 동일하게 sparse 영역은 0이므로 같은 패턴 적용.

---

## [01 Stereo Depth] Block Matching이 느린 이유 — 파이썬 루프 한계

**증상**
Block Matching (SAD) 512×256 이미지 1장에 수십 초 소요.

**원인**
Python 순수 루프로 `for d in range(max_disp)` 반복 → CUDA/NumPy 벡터화 없음.

**해결**
- 평가 목적 (Classical baseline 비교)이므로 속도보다 정확성 우선
- 실제 산업에서는 SGM(Semi-Global Matching) 사용 — OpenCV `cv2.StereoSGBM_create()`
- Learning-based(PSMNet) vs Classical(BM) 비교 목적이므로 허용 가능한 트레이드오프

**교훈**
Classical 알고리즘을 직접 구현할 때 교육 목적으로는 OK, 속도 필요하면 OpenCV/C++ 확장.

---

## [01 Stereo Depth] PSMNet-lite cost volume 메모리 — `MAX_DISP` 크기 선택

**증상**
`MAX_DISP=192`로 cost volume 생성 시 `(B, 2C, D, H/4, W/4)` 4D 텐서가 GPU 메모리 초과.

**원인**
Cost volume 크기 = B × 2C × D × H × W — D가 크면 지수적으로 증가.

**해결**
- `MAX_DISP=192` → `MAX_DISP=64`로 조정 (합성 데이터 depth range에 맞게)
- 실제 KITTI에서는 MAX_DISP=192가 표준이지만 배치 크기를 줄여 대응

**교훈**
Cost volume 기반 네트워크(PSMNet, GwcNet)는 MAX_DISP에 메모리가 선형 비례.
RTX 4080 16GB 기준: KITTI 해상도(1242×375) + MAX_DISP=192 → batch=2도 빠듯함.

---

## [02 Panoptic Seg] things 클래스 PQ가 stuff보다 낮음

**증상**
최종 PQ = 0.7499이나 클래스별 분해에서 things(car, pedestrian) PQ가 stuff(road, sky)보다 낮음.

**원인**
- **Instance 분리 어려움**: stuff는 픽셀 단위 분류(semantic)만 필요하지만, things는 같은 클래스 여러 인스턴스를 분리해야 함.
- **PQ 계산 엄격성**: things 매칭은 IoU ≥ 0.5 기준, 미매칭은 FP/FN으로 처리 → RQ가 낮아짐.
- **합성 데이터 특성**: 인스턴스가 겹치는 경우(occluded car) Instance Head가 분리 실패.

**해결 방향**
- Instance head에 더 깊은 feature 사용 (FPN 멀티스케일 feature 활용)
- 실제 모델(Mask2Former)은 Transformer query로 각 인스턴스를 독립 slot에 할당해 이 문제 해결

**교훈**
Panoptic에서 things의 낮은 PQ는 instance 분리 실패이지 semantic 분류 실패가 아님.
SQ(Segmentation Quality)가 높아도 RQ(Recognition Quality)가 낮으면 PQ는 낮아진다.

---

## [02 Panoptic Seg] 가변 크기 boxes/labels → DataLoader collate 오류

**증상**
```
RuntimeError: stack expects each tensor to be equal size
```
기본 DataLoader collate에서 배치 내 boxes 텐서 크기가 서로 달라 스택 불가.

**원인**
샘플마다 things 인스턴스 수(N)가 다름 → boxes shape이 `(N, 4)` 인데 N이 가변.

**해결**
```python
def panoptic_collate(batch):
    imgs   = torch.stack([b[0] for b in batch])   # 고정 크기
    sems   = torch.stack([b[1] for b in batch])   # 고정 크기
    insts  = torch.stack([b[2] for b in batch])   # 고정 크기
    boxes  = [b[3] for b in batch]                # 리스트로 — 스택 안 함
    labels = [b[4] for b in batch]               # 리스트로
    return imgs, sems, insts, boxes, labels
```

**교훈**
Detection/Panoptic 데이터셋에서 per-sample 객체 수가 다르면 커스텀 collate 필수.
DETR도 동일 패턴: `detr_collate`에서 boxes/labels를 리스트로 반환.

---

## [02 Panoptic Seg] panoptic ID 인코딩 충돌

**증상**
PQ 계산 시 stuff class_id와 things 인코딩 값이 겹쳐 매칭 오류.

**원인**
things 인코딩 = `class_id * 1000 + inst_id` 인데, `N_STUFF=3`, things class_id=3,4.
inst_id=0이면 things 인코딩=3000 or 4000 → stuff class_id(0~2)와 겹치지 않지만,
**inst_id=0을 배경으로 쓰면서** things_mask 처리 순서 실수 시 stuff 영역 덮어씌움.

**해결**
inst_id=0 → 배경(no instance)로 예약, things 인스턴스는 1부터 시작.
GT 구성 시:
```python
gt_pan = gt_sem.copy()                    # stuff: class_id 그대로
things_mask = gt_sem >= N_STUFF
gt_pan[things_mask] = gt_sem[things_mask] * 1000 + gt_inst[things_mask]
# gt_inst=0인 픽셀(=배경)은 things_mask에 포함되지 않도록 주의
```

**교훈**
COCO Panoptic 포맷도 동일: segment_id = cls_id * 1000 + inst_id. 0은 예약.

---

## [03 DETR] Hungarian matching — 배치 내 객체 수 불일치

**증상**
`linear_sum_assignment` 호출 시 cost matrix shape 오류.

**원인**
배치 내 각 샘플의 GT 객체 수(M)가 다른데 단순 스택하면 shape 불일치.

**해결**
```python
def detr_collate(batch):
    imgs   = torch.stack([b[0] for b in batch])
    boxes  = [b[1] for b in batch]   # 리스트 — 각 (M_i, 4)
    labels = [b[2] for b in batch]   # 리스트 — 각 (M_i,)
    return imgs, boxes, labels
```
Loss 계산 시 샘플별 루프로 Hungarian matching 수행.

**교훈**
DETR 계열 loss는 per-sample matching → 배치 병렬화가 어렵다.
실제 DETR 구현도 for loop per sample — 이게 DETR 학습이 느린 이유 중 하나.

---

## [03 DETR] GIoU loss에서 음수 IoU / NaN

**증상**
학습 초반 GIoU loss가 NaN 또는 `-inf`로 터짐.

**원인**
박스 예측값이 초기화 직후 음수이거나 w/h < 0인 경우:
- `x2 - x1` < 0 → intersection 계산에서 음수 → `.clamp(0)` 없으면 음수 넘어옴.

**해결**
```python
inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
union = area1 + area2 - inter + 1e-7   # 0 나눔 방지
```

**교훈**
GIoU/DIoU/CIoU 구현 시 `.clamp(0)` + `eps` 추가는 필수. 초기 학습 안정성에 직결.

---

## [03 DETR] Backbone vs Transformer 학습률 분리

**증상**
단일 lr로 학습 시 transformer decoder가 수렴하지 않거나 backbone이 overfit.

**원인**
Backbone(CNN)과 Transformer는 최적 lr 범위가 다름.
DETR 논문: backbone lr=1e-5, transformer lr=1e-4 (10배 차이).

**해결**
```python
param_dicts = [
    {'params': [p for n,p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5},
    {'params': [p for n,p in model.named_parameters() if 'backbone' not in n]},  # base lr
]
optimizer = torch.optim.AdamW(param_dicts, lr=1e-4, weight_decay=1e-4)
```

**교훈**
DETR 계열 논문에서 backbone lr=1/10 패턴은 거의 표준.
Deformable DETR, DN-DETR도 동일하게 적용.

---

## [공통] Windows Jupyter — `num_workers > 0` 멀티프로세싱 오류

**증상**
```
RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase
```

**원인**
Windows에서 `multiprocessing` spawn 방식 + Jupyter 환경이 충돌.

**해결**
```python
DataLoader(..., num_workers=0)  # Windows + Jupyter에서는 0 고정
```

**교훈**
Linux 서버 배포 전 Windows 개발 환경에서는 `num_workers=0`.
Linux에서는 `num_workers=4` 이상 사용 가능.

---

## 최종 결과 수치 (참고용)

| 노트북 | 모델 | 핵심 지표 |
|--------|------|-----------|
| 01_stereo_depth | PSMNet-lite vs Block Matching | EPE 1.406px (PSMNet) vs BM |
| 02_panoptic_seg | PanopticFPN (LightResNet+FPN) | PQ 0.7499, mIoU 0.9245 |
| 03_detr | DETR (CNN+Transformer) | mAP@0.5: 노트북 실행 결과 참조 |
