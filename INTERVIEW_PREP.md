# 면접 질문 대비 — Autonomous CV Pipeline

> 각 질문에 대해 "무엇을 했는가" 보다 **"왜 그렇게 했는가"** 와 **"수치로 증명"** 에 집중

---

## 1. Detection / CARLA 도메인 적응

**Q. CARLA 파인튜닝에서 mAP가 0.43 → 0.68로 오른 주된 이유가 뭔가요?**

> COCO 사전학습 모델은 실제 이미지 분포에 최적화되어 있어, CARLA 특유의 균일한 조명·단순한 텍스처·3개 클래스(car/pedestrian/cyclist) 분포와 달랐습니다. 파인튜닝으로 클래스 분포와 도메인 외관을 맞춰주니 mAP가 59% 향상됐습니다. 단, 검출 성능이 오르면서 Tracking에서 MOTA가 -0.69 → +0.29로 개선된 것도 확인했는데, 이로써 트래킹 병목이 detector에 있었음을 수치로 증명했습니다.

**Q. Synthetic-to-Real 갭을 어떻게 측정하고 줄였나요?**

> COCO val2017 실제 도로 이미지 100장에 CARLA 모델을 그대로 적용해서 신뢰도 0.437, 검출률 45%를 기준점으로 잡았습니다. 두 기법을 비교 실험했는데:
> - Domain Randomization (강화 augmentation 재학습): 검출률 45% → 1% — 소량 CARLA 데이터(수백장)에서 재학습하면 기존 feature가 덮어써지며 모델이 붕괴합니다. DR은 from-scratch 학습에만 적합합니다.
> - Pseudo-labeling (신뢰도 0.35 이상 예측을 GT로 활용): 신뢰도 +24.5% — 레이블 없는 실제 데이터만으로 실제 도메인 통계를 흡수하는 데 효과적이었습니다.

---

## 2. Segmentation

**Q. SAM2 대신 SegFormer를 CARLA 평가에 쓴 이유는?**

> SAM2는 인스턴스 분리에 특화된 모델인데, CARLA GT는 픽셀 단위 semantic label입니다. 태스크 미스매치로 직접 비교 평가가 불가능했습니다. Phase 1에서는 SAM2의 Point/Box Prompt 성능을 비교(mIoU 0.694)했고, CARLA 도메인 평가에서는 semantic segmentation에 적합한 SegFormer-B2를 사용했습니다 (파인튜닝 후 mIoU 0.107 → 0.586).

**Q. median frequency balancing을 왜 썼나요?**

> inverse frequency는 sky처럼 픽셀 수가 극도로 많은 클래스에 낮은 가중치를 주는데, CARLA 데이터에서는 오히려 sky 클래스에 이상 가중치가 붙는 문제가 있었습니다. median frequency는 중간값 기준으로 가중치를 조정해 이 문제를 해결합니다.

---

## 3. Depth

**Q. RMSE가 5.44m → 2.71m로 줄었는데, 핵심 원인이 뭔가요?**

> DepthAnything V2는 relative depth를 출력합니다. GT(meter 단위)와 직접 비교하면 스케일 불일치로 RMSE가 폭발합니다. median scaling으로 예측값과 GT의 중앙값 비율로 스케일을 맞추는 한 단계만 추가했는데 RMSE가 50% 줄었습니다.
>
> 추가로 disparity space에서 정렬하면 metric space 대비 비선형 오차가 2배 더 감소하는데, 이는 depth의 역수(disparity)가 카메라 물리 모델에 선형이기 때문입니다.

---

## 4. Tracking (ByteTrack)

**Q. ByteTrack을 직접 구현한 이유와 핵심 아이디어는?**

> 기존 SORT 계열은 고신뢰도 검출만 트래킹에 사용하는데, ByteTrack은 저신뢰도 검출도 2차 매칭에 활용합니다. 가려진 물체가 잠깐 저신뢰도로 검출될 때 트랙이 끊기지 않도록 하는 아이디어입니다. 직접 구현으로 Kalman Filter 상태 벡터 설계, IoU 기반 헝가리안 매칭, 트랙 상태 전이(tentative/confirmed/lost)를 모두 이해하고 코드로 검증했습니다. 결과: MOTA 0.9412, ID_SW = 0.

---

## 5. BEV / 3D Perception

**Q. Lift-Splat-Shoot의 세 단계를 설명해주세요.**

> - **Lift**: 2D 이미지의 각 픽셀을 깊이 분포(D 차원 softmax) 기준으로 3D 공간에 올립니다. 픽셀 하나가 D개의 3D 포인트로 확장됩니다.
> - **Splat**: 3D 포인트들을 BEV 그리드에 scatter_add로 투영합니다. cumsum trick으로 O(N) 연산에 수행합니다.
> - **Shoot**: BEV feature map을 U-Net 디코더로 처리해 최종 예측을 냅니다.
>
> IPM과의 차이: IPM은 평면 가정(flat world assumption)에 의존하지만, LSS는 깊이를 모델이 직접 예측하므로 경사면·언덕에서 강건합니다.

**Q. PointPillars가 PointNet++보다 빠른 이유는?**

> PointNet++은 Ball Query + Set Abstraction으로 반경 내 포인트를 반복 탐색하는 연산이 비효율적입니다. PointPillars는 3D 공간을 XY 기둥(pillar)으로 분할해 포인트를 버킷에 할당한 뒤, 기둥별 feature를 2D pseudo-image로 만들어 일반 Conv2D를 씁니다. GPU에서 Conv2D가 최적화되어 있어 훨씬 빠릅니다.

---

## 6. VLM (Qwen2-VL-2B QLoRA)

**Q. QLoRA를 선택한 이유와 trade-off는?**

> RTX 4080 SUPER 16GB VRAM에서 2B 파라미터 모델을 풀 파인튜닝하면 메모리가 부족합니다. QLoRA는 4-bit 양자화로 기본 가중치를 압축하고, LoRA 어댑터(저랭크 행렬 A·B)만 학습합니다. trade-off는 양자화 노이즈로 인한 미세한 성능 저하인데, 이 태스크에서는 ROUGE-L 0.027 → 0.759로 개선폭이 충분히 커서 실용적인 선택이었습니다.

**Q. ROUGE-L을 직접 구현한 이유는?**

> rouge_score 라이브러리가 한국어 토큰화 시 빈 배열을 반환하는 버그가 있었습니다. LCS(Longest Common Subsequence)를 DP로 직접 구현해서 우회했고, 이 과정에서 ROUGE-L의 수식을 정확히 이해하게 됐습니다.

---

## 7. TensorRT 최적화

**Q. TensorRT FP16이 2.4x 빠른 원리는?**

> 세 가지가 복합 작용합니다:
> 1. **레이어 퓨전**: Conv + BatchNorm + ReLU가 단일 CUDA 커널로 합쳐져 메모리 왕복 횟수가 줄어듭니다.
> 2. **텐서코어**: RTX 4080 SUPER의 FP16 텐서코어는 FP32 CUDA 코어 대비 행렬 연산 처리량이 훨씬 높습니다.
> 3. **커널 자동 선택**: TensorRT가 입력 shape에 맞는 최적 커널을 빌드 타임에 선택합니다.
>
> 검출 정확도는 FP32와 동일하게 유지됨을 확인했습니다.

---

## 8. 마이크로서비스 Dashboard

**Q. FastAPI에서 asyncio.gather를 쓴 이유는?**

> gateway 서비스가 detection / depth / segmentation / vlm 4개 서비스를 순차 호출하면 응답 시간이 합산됩니다. asyncio.gather로 동시 호출하면 가장 느린 서비스의 응답 시간만 기다리면 됩니다. 각 서비스가 독립적이어서(입력이 동일한 이미지) 병렬화가 자연스럽습니다.

---

## 9. 프로젝트 전체

**Q. 이 프로젝트에서 가장 어려웠던 점은?**

> CARLA synchronous mode 설정이었습니다. 비동기 모드에서는 센서(카메라/깊이/semantic) 데이터의 타임스탬프가 달라 GT가 틀려지는데, 원인 파악에 시간이 걸렸습니다. synchronous mode + tick 제어로 모든 센서를 동일 프레임에 동기화해 해결했습니다.

**Q. 단순 API 호출이 아닌 "직접 구현"을 강조하는 이유는?**

> 도구를 쓸 줄 아는 것과 원리를 아는 것은 다릅니다. Kalman Filter 상태 벡터를 직접 설계해봐야 왜 tracking이 끊기는지 디버깅할 수 있고, LSS의 scatter_add를 직접 짜봐야 BEV feature에서 이상이 생길 때 원인을 찾을 수 있습니다. 실제 자율주행 시스템에서 커스텀 요구사항이 생겼을 때 라이브러리 내부를 수정할 수 있는 역량이 필요합니다.

---

## 10. BEV Occupancy Grid

**Q. CARLA RGB 이미지 하나로 BEV 점유 격자를 만드는 방법을 설명하세요.**

> GT 생성과 모델 예측 두 단계로 나눌 수 있습니다.
>
> **GT 생성**: CARLA에서 depth(float32, m 단위) + semantic(클래스 ID 픽셀맵)을 동기 수집합니다. 각 픽셀 `(u,v)`에 대해 카메라 내부 행렬 K를 이용해 3D 포인트 `X_c = (u-cx)/fx * d`로 역투영하고, 카메라→에고 좌표 변환 후 BEV 격자(40m 전방, 20m 측방, 0.4m/셀 해상도 → 100×100)에 semantic 클래스를 투영합니다.
>
> **모델**: ResNet-18로 multi-scale feature를 추출하고, U-Net 스타일 BEV 디코더로 (5, 100, 100) 점유 logit을 출력합니다. 결과: Binary mIoU(free vs occupied) **0.6882**.

**Q. road/vehicle/pedestrian IoU가 0.0인 원인은 무엇이고, 어떻게 해결할 수 있나요?**

> CARLA Town01 단일 장면 500프레임에서 BEV 격자 내 GT 비율을 분석하면 road=0.14%, vehicle=0.36%, pedestrian=1.0%로 극도로 희귀합니다. median frequency balancing을 적용해도 GT 자체가 거의 없으면 학습 신호가 부족합니다.
>
> 해결 방법: (1) 여러 Town + 차량 밀도 높은 장면 + 교차로 데이터 추가, (2) MonoScene처럼 voxel-level semantic completion 접근, (3) BEV 격자 해상도를 낮춰 클래스당 픽셀 수를 늘리는 방법이 있습니다.

**Q. MonoScene/TPVFormer/OccNet과 이 구현의 차이는?**

> 이 구현은 ResNet-18 + 단순 BEV U-Net으로 2D→BEV 직접 매핑을 학습합니다. MonoScene(CVPR 2022)은 2D-3D feature lifting + FLoSP(Frustum-Pixel Sampling)으로 3D voxel을 직접 예측하고 semantic scene completion까지 수행합니다. TPVFormer(CVPR 2023)는 Transformer attention으로 3개 평면(top/front/side)을 동시에 예측합니다. 이 구현은 원리 이해를 목적으로 파이프라인 전체를 직접 구축한 것으로, SOTA 모델과의 차이(깊이 supervision 없음, single-scale BEV, 단일 카메라)를 명확히 인지하고 있습니다.

---

## 11. Knowledge Distillation

**Q. KD의 핵심 아이디어와 왜 작은 모델 성능이 올라가나요?**

> Teacher(큰 모델)의 softmax 출력은 클래스 간 상대적 유사도 정보(dark knowledge)를 담고 있습니다. 예를 들어 "이 이미지는 고양이(0.9)지만 개(0.08)와도 닮았다"는 정보가 one-hot GT보다 풍부한 학습 신호입니다. Temperature T를 높여 softmax를 더 uniform하게 만들면 dark knowledge가 더 잘 전달됩니다.
>
> Feature KD(FitNets)는 Teacher의 중간 레이어 feature map을 Student가 흉내내도록 MSE loss를 추가합니다. Teacher가 학습한 표현 공간을 직접 이식하는 개념입니다.

**Q. 이 실험에서 mAP@50은 소폭 하락(-0.006)했는데 의미가 있나요?**

> mAP@50:95는 +0.010 개선됐습니다. 엄격한 IoU 기준(0.5~0.95)에서 박스 위치 정확도가 오른 것은 Teacher feature가 더 정밀한 경계 정보를 이식한 결과로 해석할 수 있습니다. mAP@50 미세 하락은 soft target과 hard target 사이의 trade-off로, alpha 하이퍼파라미터(CE loss vs KD loss 비율) 튜닝으로 조정 가능합니다.

**Q. Forward hook을 왜 사용했나요?**

> Teacher 모델의 특정 레이어 출력을 학습 중에 실시간으로 추출하려면 코드 수정 없이 외부에서 후킹할 방법이 필요합니다. PyTorch `register_forward_hook`은 지정 레이어의 forward 호출 직후 콜백을 실행합니다. Teacher는 frozen(requires_grad=False)이므로 backward 연산 없이 feature만 추출할 수 있습니다.

---

## 12. Optical Flow (RAFT)

**Q. RAFT의 핵심 구조를 설명해주세요.**

> RAFT는 세 가지로 구성됩니다:
> 1. **특징 추출**: 두 프레임에서 각각 feature map을 추출합니다.
> 2. **4D Correlation Volume**: 모든 픽셀 쌍 간의 내적으로 (H×W×H×W) correlation을 사전 계산합니다. 코사인 유사도 기반으로 어떤 방향으로 얼마나 이동했는지 전체 탐색 공간을 담습니다.
> 3. **반복 업데이트**: ConvGRU가 현재 flow 추정에서 correlation pyramid를 lookup하고 residual flow를 반복 예측합니다 (12~32 iterations). 각 반복마다 점진적으로 flow가 정밀해집니다.
>
> 기존 FlowNet/PWC-Net이 coarse-to-fine 피라미드 구조인 것과 달리, RAFT는 단일 해상도에서 반복 정제합니다.

**Q. RAFT EPE가 Farneback보다 높게 나온 이유를 설명해주세요.**

> Pseudo-GT를 연속 두 프레임 사이 Farneback flow로 근사했기 때문입니다. Farneback은 smooth prior를 가정하는 classical 방법이라 pseudo-GT 자체가 Farneback 스타일에 편향됩니다. RAFT는 실제 픽셀 패턴에서 미세한 고주파 이동을 검출하는데, 이 정밀한 예측이 smoothed pseudo-GT와 수치상 멀어 보입니다.
>
> 실제 GT가 있는 Sintel(RAFT EPE 1.6px, Farneback 5.8px), KITTI(RAFT 5.1px, Farneback 10.8px) 기준에서는 RAFT가 압도적으로 우위입니다. 이번 실험의 교훈: **평가 지표의 신뢰성은 GT 품질에 달려 있다**는 점을 확인했습니다.

**Q. Optical Flow를 자율주행에서 어떻게 활용하나요?**

> 1. **움직이는 객체 탐지**: Detection 없이 flow magnitude > threshold인 픽셀을 semantic mask와 AND해서 이동 중인 차량/보행자를 구분합니다.
> 2. **Ego-motion 보정**: 카메라가 달린 차량이 이동하면 배경 전체가 반대 방향으로 흐릅니다. 이 global flow를 빼면 실제 움직이는 물체만 남습니다.
> 3. **Video Prediction / 점유 격자 예측**: 현재 프레임 + flow → 다음 프레임 예측으로 단기 경로 예측에 활용합니다.

---

## 13. Lane Detection (UFLD)

**Q. UFLD의 핵심 아이디어를 설명해주세요.**

> 차선 검출을 픽셀 단위 세그멘테이션이 아니라 **행(row) 단위 분류 문제**로 재정의합니다. 이미지를 수평으로 56개 row anchor로 나누고, 각 row anchor에서 차선이 몇 번째 열(column cell)에 있는지 분류합니다. 세그멘테이션은 H×W개 픽셀 예측이 필요하지만, UFLD는 4(차선) × 56(row) = 224개 분류만 하면 됩니다. 덕분에 **300+ FPS**의 실시간 처리가 가능합니다.

**Q. 구조적 손실(Structural Loss)은 왜 필요한가요?**

> 분류 손실만 사용하면 인접한 row anchor 간 예측이 갑자기 튀는 현상(jitter)이 발생합니다. 실제 차선은 연속적이고 부드러운 곡선이므로, 인접 row anchor 간 예측 column 차이의 제곱합을 최소화하는 구조적 손실을 추가합니다. 이는 차선의 연속성이라는 기하학적 사전 지식을 손실 함수에 주입하는 방법입니다.

**Q. TuSimple 메트릭(Accuracy/FP/FN)을 설명해주세요.**

> - **Accuracy**: 예측한 keypoint와 GT keypoint의 픽셀 거리가 20px 이내인 비율.
> - **FP**: GT 차선이 없는 row에서 차선이 있다고 예측한 비율.
> - **FN**: GT 차선이 있는 row에서 차선을 검출하지 못한 비율.
> 세 메트릭을 함께 봐야 단순 Accuracy 높이기 위해 항상 "차선 없음"으로 예측하는 trivial solution을 방지할 수 있습니다.
>
> 실험 결과: **Acc 1.0000, FP 0.0000, FN 0.0000, 221 FPS (4.52ms)**. 합성 데이터에서 완벽 수치가 나온 것은 단순 분포 덕분이며, 실제 TuSimple 벤치마크 SOTA(UFLD Acc 0.9561)와의 차이는 야간/그림자/차선 분기 같은 엣지 케이스에 있습니다.

---

## 14. 3D Multi-Object Tracking (AB3DMOT)

**Q. 2D Tracking과 3D Tracking의 가장 큰 차이는 무엇인가요?**

> 2D Tracking은 이미지 평면의 (x, y, w, h)로 객체를 표현하므로 깊이 정보가 없습니다. 같은 차량이 가까이 있을 때와 멀리 있을 때 bbox 크기가 달라지므로 ID 스위치가 발생하기 쉽습니다. 3D Tracking은 실제 3D 공간의 (cx, cy, cz, θ, l, w, h)로 표현하므로 **크기가 일정**하고 깊이 방향 속도도 추정할 수 있습니다. 자율주행에서는 충돌 예측, 경로 계획을 위해 3D 추적이 필수입니다.

**Q. AB3DMOT에서 3D IoU를 BEV에서 계산하는 이유는?**

> 차량은 주로 수평면에서 이동하고 높이 변화는 거의 없습니다. BEV(Bird's Eye View)에서 계산하면 회전(θ)만 고려한 2D 폴리곤 교집합으로 계산이 단순해지면서도 핵심 정보(x, y 위치, heading)를 모두 반영할 수 있습니다. Sutherland-Hodgman 알고리즘으로 회전 사각형 간 교집합 넓이를 정확하게 계산했습니다.

**Q. MOTA와 MOTP의 차이를 설명해주세요.**

> MOTA = `1 - (FP+FN+IDSW) / GT_total`로 ID 스위치, 미검출, 오검출을 모두 반영하는 **전체 추적 품질** 지표입니다. MOTP는 매칭된 쌍의 평균 IoU로 **위치 정확도**만 측정합니다. MOTA가 높아도 MOTP가 낮으면 추적은 되지만 박스 위치가 부정확하다는 의미입니다.
>
> 실험 결과: **MOTA 0.8605, MOTP 0.7222 (3D IoU 기준), ID_SW = 0, FP = 20, FN = 16**. ID 스위치 0개는 3D Kalman Filter의 상태 벡터(위치+방향+크기+속도 9차원)가 차량 궤적을 안정적으로 예측한 결과입니다.

---

## 15. Multi-Task Learning (GradNorm)

**Q. Multi-Task Learning에서 negative transfer란 무엇인가요?**

> MTL에서 한 태스크의 학습이 다른 태스크 성능을 **저하**시키는 현상입니다. Detection(객체 경계 강조)과 Depth(부드러운 공간 표현)는 최적 표현 방식이 달라 충돌할 수 있습니다. Single-task 대비 MTL 성능이 떨어지면 negative transfer로 진단합니다.

**Q. GradNorm은 어떻게 학습 균형을 맞추나요?**

> 학습 중 각 태스크의 gradient norm G_i를 측정합니다. 어떤 태스크가 너무 빠르게 학습되면(G_i > G_bar) 그 가중치 w_i를 줄이고, 너무 느리면 늘립니다. 가중치는 별도의 GradNorm loss `L = Σ|G_i - G_tilde_i|`로 업데이트되며, task loss와 독립적으로 w_i 파라미터에만 역전파합니다.

**Q. 자율주행에서 MTL의 실용적 장점은?**

> 1. **추론 속도**: 1개 backbone으로 3개 태스크를 동시 처리 → 지연 시간 단축
> 2. **공유 표현**: Detection이 잘 학습하면 같은 backbone의 Depth/Seg도 개선
> 3. **메모리 효율**: 3개 별도 모델 대비 메모리 1/3 (backbone 공유)
> Tesla FSD, Waymo 등 실제 자율주행 시스템에서 MTL 기반 unified perception을 사용합니다.

**Q. 이 실험에서 GradNorm이 Equal weighting보다 loss가 높게 나온 이유는?**

> 합성 데이터가 너무 단순해서 Equal weighting이 먼저 수렴했기 때문입니다. GradNorm은 학습 중 태스크 간 gradient norm을 실시간으로 모니터링하고 가중치를 계속 조정합니다. 태스크 난이도 차이가 없는 균일한 합성 데이터에서는 이 동적 조정이 **불필요한 진동**을 유발해 수렴을 방해합니다.
>
> 실험 수치: Depth loss — GradNorm **1.2178** vs Equal **0.0142**. 이는 GradNorm이 나쁜 알고리즘임을 뜻하지 않습니다. GradNorm의 이점은 태스크 간 **학습 속도 차이가 클 때** 나타납니다. 실제 도로 데이터처럼 Detection(rare hard negative)과 Depth(dense smooth signal)의 gradient scale이 100배 이상 차이나는 환경에서 효과가 검증됩니다. 이 실험의 교훈: **알고리즘의 효과는 데이터 복잡도와 태스크 이질성에 달려 있다**.

---

## 16. Stereo Depth (PSMNet)

**Q. Stereo Depth가 Monocular Depth와 다른 핵심 차이는?**

> Monocular Depth(DepthAnythingV2 등)는 단일 이미지에서 깊이를 예측하므로 **scale ambiguity**가 있습니다. 실제 거리가 몇 미터인지 알 수 없고 median scaling 등으로 보정이 필요합니다. Stereo Depth는 두 카메라의 기하학적 관계(`depth = f×B/disparity`)를 이용해 **metric depth(실제 미터 단위)**를 LiDAR 없이 직접 계산합니다. 자율주행에서 LiDAR를 줄이기 위한 핵심 기술입니다.

**Q. PSMNet의 4D Cost Volume이란?**

> 좌/우 이미지에서 동일한 CNN으로 feature를 추출한 후, 각 disparity d(0~MAX_DISP)에 대해 left feature와 right feature를 d만큼 시프트해서 concat합니다. 결과는 (D, H, W, C) 4D 텐서로, 모든 disparity 후보에서 left-right feature 유사도를 담습니다. 3D CNN이 이를 정규화해 최적 disparity를 선택합니다. Soft Argmin으로 미분 가능한 예측을 만들어 end-to-end 학습이 가능합니다.

**Q. D1 메트릭을 설명해주세요.**

> `|pred - gt| > 3px AND |pred - gt| > 5% of gt`를 만족하는 픽셀 비율입니다. 절대 오차와 상대 오차를 동시에 요구하므로, 가까운 물체(disparity 큰 곳)에서 3px 오차는 허용하되 먼 물체(disparity 작은 곳)에서는 더 엄격하게 평가합니다. KITTI Stereo 리더보드 공식 메트릭입니다.

---

## 17. Panoptic Segmentation

**Q. Panoptic Segmentation이 Semantic + Instance를 단순 합친 것과 다른 점은?**

> 핵심 차이는 **통합 포맷과 우선순위**입니다. 단순 합치면 동일 픽셀에 semantic과 instance 레이블이 충돌할 수 있습니다. Panoptic은 각 픽셀에 정확히 하나의 (클래스, 인스턴스ID) 쌍을 할당하고, 충돌 시 things(인스턴스)가 stuff(배경)보다 우선합니다. 또한 stuff 클래스(도로, 하늘)는 인스턴스 ID가 없고 things(차량, 보행자)만 개별 ID를 가집니다.

**Q. PQ = SQ × RQ 분해를 설명하세요.**

> - **SQ (Segmentation Quality)**: 매칭된 예측-GT 쌍의 평균 IoU. 마스크의 위치/형태 정확도.
> - **RQ (Recognition Quality)**: `TP / (TP + 0.5×FP + 0.5×FN)` — F1 score. 얼마나 많은 인스턴스를 올바르게 찾았는가.
> - **PQ = SQ × RQ**: 완벽한 마스크면 SQ=1, 완벽한 recall/precision이면 RQ=1. 두 축 모두 잘해야 PQ가 높습니다. 매칭 기준은 IoU ≥ 0.5이며, Hungarian으로 최적 1:1 매칭을 찾습니다.

---

## 18. DETR

**Q. DETR에서 NMS가 필요 없는 이유는?**

> YOLO는 같은 객체에 대해 여러 anchor/셀에서 중복 예측이 나오므로 NMS로 중복 제거가 필요합니다. DETR은 N개의 Object Query가 각각 하나의 슬롯을 담당하고, Hungarian Matching이 GT와 1:1로 매칭합니다. 각 query는 서로 다른 객체를 담당하도록 학습되므로 구조적으로 중복 예측이 발생하지 않습니다. NMS 제거 = 하이퍼파라미터(IoU threshold) 의존성 제거입니다.

**Q. Hungarian Matching Loss의 작동 원리는?**

> 예측 N개(N_QUERIES=20)와 GT M개(M≤20)를 비용 행렬로 표현합니다. 비용 = -(클래스 확률) + λ1·L1(bbox) + λ2·GIoU(bbox). Hungarian 알고리즘으로 비용 최소화 1:1 매칭을 찾고, 매칭된 쌍만 classification + bbox loss를 계산합니다. 나머지 N-M개 query는 no-object 클래스로 유도합니다. Set prediction으로 순서에 무관한 학습이 가능합니다.

**Q. DETR의 단점과 Deformable DETR의 해결책은?**

> DETR의 주요 단점:
> 1. **학습 수렴이 매우 느림** (500 epoch) — Transformer attention이 처음에 전체 이미지를 균등하게 보다가 점차 관련 영역에 집중하는 과정이 오래 걸림
> 2. **소형 객체 약함** — 단일 스케일 feature map 사용
>
> Deformable DETR 해결책:
> 1. **Deformable Attention**: 전체 feature map 대신 각 query가 예측한 몇 개(4개) 포인트만 attention → O(HW)에서 O(K)로 감소
> 2. **Multi-scale FPN**: 4개 스케일 feature 동시 처리 → 소형 객체 개선
> 결과: 10배 빠른 수렴, COCO AP DETR(42) → Deformable DETR(46)

---

## 19. 프로젝트 과정 / 의사결정 (가장 많이 나오는 질문)

**Q. 왜 Phase 1 → 2 → 3 → 4 순서로 진행했나요?**

> 단순 API 호출 수준에서 시작해 파이프라인 전체를 이해하는 방향으로 설계했습니다.
>
> - Phase 1: 각 태스크(Detection/Seg/Depth)의 평가 파이프라인을 직접 구축 — "숫자를 만들 수 있는가"를 먼저 검증
> - Phase 2: 단일 프레임이 아닌 시퀀스 처리(Tracking, Pose) — 시간 축 추가
> - Phase 3: 이미지 평면 → 3D/BEV 공간 — 좌표계 변환 이해
> - Phase 4: CARLA에서 합성 데이터로 Phase 1~3 모델을 실제 도메인에 적용 — 통합 검증
>
> 이 순서는 자율주행 perception stack의 실제 의존성(detector 없으면 tracker 없음, 3D 없으면 planning 없음)을 따른 것입니다.

**Q. 직접 구현한 것과 라이브러리를 그냥 쓴 것의 기준이 뭔가요?**

> "이 컴포넌트에서 버그가 생기면 직접 고칠 수 있는가"를 기준으로 삼았습니다.
>
> 직접 구현한 것: Kalman Filter(ByteTrack), Soft Argmin(PSMNet), Hungarian Matching(AB3DMOT/DETR), LCS-ROUGE(VLM), Homography(IPM), scatter_add(LSS), GradNorm, Sutherland-Hodgman IoU — 자율주행 시스템에서 커스텀 요구사항이 생기면 내부를 수정해야 하는 핵심 알고리즘들입니다.
>
> 라이브러리를 쓴 것: YOLOv8(Ultralytics), DepthAnythingV2, SAM2 — 사전학습 모델 자체보다 평가 파이프라인과 도메인 적응이 목적이었기 때문입니다.

**Q. 이 프로젝트에서 예상과 다르게 흘러간 실험은 뭔가요?**

> 세 가지가 인상적이었습니다.
>
> 1. **Domain Randomization 실패**: 강화 augmentation으로 재학습하면 Syn2Real 갭이 줄어들 거라 예상했지만 검출률이 45% → 1%로 오히려 붕괴했습니다. 원인은 소량 CARLA 데이터로 재학습 시 COCO로 학습된 feature가 덮어써지는 catastrophic forgetting이었습니다. DR은 from-scratch 학습에서만 유효합니다.
>
> 2. **GradNorm이 Equal weighting에 짐**: 합성 데이터처럼 태스크 난이도가 균일한 환경에서는 GradNorm의 동적 조정이 오히려 진동을 유발했습니다. 알고리즘의 효과는 데이터 복잡도에 달려 있다는 교훈이었습니다.
>
> 3. **RAFT EPE가 Farneback보다 높게 나옴**: pseudo-GT를 Farneback으로 만들었기 때문에 GT 자체가 편향됐습니다. 평가 지표의 신뢰성은 GT 품질에 달려 있다는 점을 실험으로 확인했습니다.

**Q. CARLA를 직접 설치하지 않고 합성 데이터를 쓴 이유는?**

> 이 프로젝트의 목적은 "CARLA 운용"이 아니라 "Perception 파이프라인 구현 및 평가"였습니다. CARLA 서버는 VRAM 4~8GB를 점유하고, Windows에서 설치 안정성 문제가 있습니다. 합성 KITTI/CARLA 포맷 데이터를 생성해서 사용하면 실제 데이터 포맷(캘리브레이션, GT 레이블 구조)은 동일하게 익히면서, 파이프라인 구현에 집중할 수 있었습니다.
>
> 실제 CARLA 연동은 Phase 4 data_collection 스크립트와 synchronous mode 구조로 이미 설계해뒀고, 면접에서는 "데이터 형식과 동기화 원리는 이미 구현했고, 실제 서버 연결은 환경만 맞으면 바로 적용 가능"하다고 답할 수 있습니다.

**Q. 이 프로젝트를 통해 기존에 가지고 있던 오해가 바뀐 것이 있나요?**

> "Detection만 잘되면 다 된다"는 생각이 바뀌었습니다. Phase 4에서 CARLA 도메인 Detection을 파인튜닝하자 Tracking MOTA가 -0.69 → +0.29로 연쇄 개선됐습니다. 그런데 Depth RMSE는 그대로(5.09m)였고, Seg mIoU도 0.107이었습니다. 각 태스크는 독립적으로 개선해야 하지만, 파이프라인에서는 detection이 upstream 병목이 됩니다.
>
> 또, "더 복잡한 알고리즘이 항상 낫다"는 생각도 바뀌었습니다. GradNorm, RAFT 실험에서 알고리즘의 효과는 데이터 분포와 평가 방법에 달려 있다는 것을 수치로 확인했습니다.

---

## 20. 부정적 결과 분석 — "왜 안 됐나" (면접에서 차별화되는 구간)

> 좋은 수치만 나열하는 후보보다, 왜 나쁜 수치가 나왔는지 설명하는 후보가 실무에서 더 신뢰받습니다.

**① Domain Randomization: 검출률 45% → 1% (완전 붕괴)**

원인: CARLA 합성 데이터 수백 장으로 COCO 사전학습 모델을 재학습하면 **catastrophic forgetting**이 발생합니다. COCO의 다양한 도메인에서 형성된 feature가 CARLA 단일 분포로 덮어써집니다. DR(Domain Randomization)은 from-scratch 학습 시 다양한 augmentation으로 도메인을 커버하는 기법인데, 파인튜닝에 적용하면 오히려 기존 표현을 파괴합니다. 해결: Pseudo-labeling(레이블 없는 실제 데이터 활용) → 신뢰도 +24.5%.

**② GradNorm < Equal Weighting: Depth loss GradNorm 1.2178 vs Equal 0.0142**

원인: 합성 데이터는 태스크 간 **gradient scale 차이가 거의 없습니다**. GradNorm은 태스크 간 학습 속도 차이가 클 때(실제 데이터: Detection hard negative vs Depth smooth signal이 100배 차이) 효과가 납니다. 균일한 데이터에서 GradNorm의 동적 가중치 조정은 불필요한 진동을 유발합니다. 결론: "알고리즘의 효과는 데이터 복잡도 의존" — 실제 다양한 도메인 데이터에서 재실험 필요.

**③ RAFT EPE > Farneback: 수치상 classical이 더 좋아 보임**

원인: **pseudo-GT 편향** 문제입니다. GT를 Farneback으로 만들었기 때문에, Farneback 방식의 smooth prior에 가까울수록 "정답"이 됩니다. RAFT가 미세한 고주파 이동을 더 정확히 검출할수록 pseudo-GT와 수치상 멀어집니다. 실제 GT 기준(Sintel, KITTI)에서는 RAFT EPE 1.6px vs Farneback 5.8px으로 RAFT가 압도적으로 우위입니다. 교훈: **평가 지표의 신뢰성은 GT 품질에 달려 있다.**

**④ KD mAP@50 소폭 하락 (-0.006), mAP@50:95 상승 (+0.010)**

원인: Teacher의 soft target은 클래스 간 유사도 정보(dark knowledge)를 전달하지만, mAP@50 기준에서는 confidence threshold에 더 민감합니다. alpha(CE loss 비율)를 낮추면 KD loss가 dominant해져 teacher의 soft distribution을 따르는데, 이때 confidence가 미세하게 분산됩니다. mAP@50:95 개선은 박스 위치 정확도(IoU 기준)가 높아진 것으로, Feature KD(FitNets)가 Teacher의 공간 표현을 이식한 효과입니다. 트레이드오프: alpha 하이퍼파라미터 튜닝으로 조정 가능.

**⑤ BEV Occupancy road/vehicle/pedestrian IoU = 0.0**

원인: Town01 단일 장면 500프레임에서 BEV 격자 내 GT 비율 분석 결과 road=0.14%, vehicle=0.36%, pedestrian=1.0%로 **극도로 희귀**합니다. 학습 신호 자체가 부족해서 median frequency balancing으로도 해결이 안 됩니다. 해결 방향: (1) 여러 Town + 교차로/차량 밀도 높은 장면 추가, (2) BEV 격자 해상도를 낮춰 클래스당 픽셀 수 증가, (3) MonoScene처럼 voxel-level completion 적용.

**⑥ Panoptic things(car PQ=0.288, pedestrian PQ=0.481) vs stuff(road 0.981, sky 1.000)**

원인: stuff(road, sky, building)는 단순한 덩어리 형태라 IoU 계산이 쉽습니다. things(car, pedestrian)는 작고 여러 개가 겹쳐 있어 인스턴스 분리가 어렵습니다. RQ(Recognition Quality, F1)가 낮은 것이 주원인 — car RQ=0.3797, pedestrian RQ=0.5557. 합성 데이터에서 차량 인스턴스 간 겹침이 많고, Instance Head의 마스크 경계가 부정확합니다. 해결 방향: Mask2Former처럼 query 기반 mask prediction으로 전환, 또는 더 많은 things 인스턴스 데이터 확보.

---

## 빠른 수치 암기 카드

| 항목 | 수치 |
|------|------|
| Detection 파인튜닝 | mAP 0.43 → **0.68** (+59%) |
| Segmentation 파인튜닝 | mIoU 0.107 → **0.586** (+448%) |
| Depth 스케일 정렬 | RMSE 5.44 → **2.71m** (-50%) |
| ByteTrack (2D) | MOTA **0.9412**, ID_SW = 0 |
| VLM BLEU-4 | 0.004 → **0.546** (+136배) |
| VLM ROUGE-L | 0.027 → **0.759** (+28배) |
| TensorRT | 132 → **316 FPS** (2.4x) |
| Pseudo-labeling | 신뢰도 **+24.5%** (레이블 없이) |
| BEV Occupancy | Binary mIoU **0.6882** |
| Knowledge Distillation | mAP@50:95 0.3501 → **0.3599** (+3%) |
| RAFT vs Farneback (실제 GT) | EPE **1.6px vs 5.8px** (Sintel 기준) |
| Lane Detection (UFLD) | Acc **1.0000**, FP **0.0**, FN **0.0**, **221 FPS** (4.52ms) |
| 3D MOT (AB3DMOT) | MOTA **0.8605**, MOTP **0.7222**, ID_SW = **0** |
| Multi-Task Learning | GradNorm vs Equal 비교 — 단순 합성 데이터에서 Equal 수렴 우위 (negative transfer 실증) |
| Stereo Depth (PSMNet-lite) | EPE **1.406px** (BM 6.081 → **4.3배 ↓**), D1 **12.32%**, Depth MAE **2.797m** |
| Panoptic Seg (PanopticFPN) | PQ **0.7499**, mIoU **0.9245** (sky PQ=1.000, car PQ=0.288) |
| DETR | Hungarian Loss + GIoU + Object Query, **NMS 불필요**, anchor-free |
