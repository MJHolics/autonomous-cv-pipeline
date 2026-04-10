# 회사별 JD 대응 가이드

> RESUME_BULLETS.md + INTERVIEW_PREP.md 기반
> 각 회사 JD에서 겹치는 키워드를 강조하는 방식으로 조합

---

## 지원 체크리스트

지원 전 반드시 확인:
- [ ] GitHub README 최신 수치 반영됐는지 확인
- [ ] 노트북 실행 결과물(플롯/출력)이 저장됐는지 확인
- [ ] 자기소개서에 회사 JD 키워드가 3개 이상 반영됐는지 확인
- [ ] 면접 준비: 해당 회사 강조 항목 3개를 1분 안에 설명할 수 있는지 연습

---

## 자기소개서 공통 템플릿

### 지원 동기 (300자 내외)

```
자율주행 Perception 파이프라인을 처음부터 끝까지 직접 구현해보고 싶었습니다.
단순히 라이브러리를 가져다 쓰는 것이 아니라, Kalman Filter 상태 벡터 설계부터
BEV 좌표 변환, Hungarian Matching까지 논문 원리를 코드로 검증하는 방식으로 진행했습니다.
[회사명]에서 [해당 업무]를 통해 이 파이프라인을 실제 도메인에 적용하고 싶습니다.
```

### 핵심 역량 (500자 내외)

```
CARLA 시뮬레이터 기반 자율주행 Perception 파이프라인을 처음부터 구축했습니다.

Detection 분야에서는 YOLOv8을 CARLA 합성 도메인에 파인튜닝해 mAP@50을 0.43에서
0.68(+59%)로 향상시켰고, Detection 개선이 Tracking MOTA를 -0.69에서 +0.29로
연쇄 개선시키는 것을 수치로 확인했습니다.

직접 구현한 알고리즘 목록: ByteTrack(Kalman Filter + 이중 매칭), PSMNet-lite
(4D Cost Volume), AB3DMOT(3D Kalman + BEV 회전 IoU), DETR(Hungarian Matching
+ GIoU), PanopticFPN(PQ 평가 파이프라인), UFLD(Row-anchor 분류), GradNorm(MTL),
LCS-ROUGE(VLM 평가) 등 핵심 알고리즘 14개 이상.

부정적 결과도 정량 분석했습니다: Domain Randomization 실패(검출률 45%→1%)를
Catastrophic Forgetting으로 분석했고, GradNorm이 Equal weighting에 진 이유를
데이터 복잡도 의존성으로 설명했습니다. 숫자로 증명하고 원인을 설명하는 것이
제 강점입니다.
```

---

## 회사 유형별 강조 전략

---

### A. 자율주행 스타트업 (라이드플럭스, 오토노머스a2z, 42dot 등)

**JD 키워드**: CARLA/시뮬레이션, Perception, 데이터 파이프라인, BEV, Detection/Tracking

**강조할 것 (우선순위 순)**

1. **CARLA 파이프라인 end-to-end**
   > "CARLA synchronous mode 기반 데이터 수집 → 파인튜닝 → 통합 평가"를 직접 구현.
   > Detection(mAP +59%), Depth(RMSE -50%), Seg(mIoU +448%), Tracking(-0.69→+0.29) 전 분야 수치 보유.

2. **BEV 구현 경험**
   > IPM(Homography 직접 구현), Lift-Splat-Shoot(NeurIPS 2020 논문 구현), nuScenes 공식 리더보드 파이프라인.

3. **실험 설계 능력 — 부정적 결과 분석**
   > DR 실패 원인(Catastrophic Forgetting) 정량 분석, Pseudo-labeling으로 전환해 +24.5% 개선.

**자기소개 첫 문장 예시**
> "CARLA 시뮬레이터 기반 멀티태스크 Perception 파이프라인을 직접 구축하고, Detection·Depth·Tracking 전 분야에서 도메인 적응 파인튜닝 수치를 검증했습니다."

---

### B. 반도체 / 엣지 AI (NVIDIA, 삼성전자, SK하이닉스, 스트라드비전)

**JD 키워드**: TensorRT, ONNX, 추론 최적화, FPS, 모델 경량화, 양자화

**강조할 것 (우선순위 순)**

1. **TensorRT FP16 최적화**
   > PyTorch → ONNX → TensorRT FP16 변환 파이프라인 직접 구축.
   > YOLOv8 추론 속도 **132 → 316 FPS** (2.4배), 검출 정확도 동일 확인.

2. **Knowledge Distillation**
   > Hinton 2015 + FitNets 논문 직접 구현.
   > YOLOv8l(Teacher, 43.7M) → YOLOv8n(Student, 3.2M): mAP@50:95 +3%, 13.7배 파라미터 감소.

3. **NLP/LLM 기반 모델 최적화 경험 (기존 강점 연결)**
   > QLoRA 4-bit 양자화로 Qwen2-VL-7B급 모델을 16GB VRAM에서 학습.
   > ONNX, 양자화 경험이 이미 있으며 CV 영역으로 확장 적용.

**자기소개 첫 문장 예시**
> "YOLOv8 TensorRT FP16 최적화로 132→316 FPS를 달성하고, Knowledge Distillation으로 모델을 13.7배 경량화하면서 mAP를 유지했습니다."

---

### C. 로보틱스 (현대로보틱스, 네이버랩스, 뉴빌리티, 서울로보틱스)

**JD 키워드**: Depth, 3D Perception, 포인트클라우드, Optical Flow, Tracking, Sensor Fusion

**강조할 것 (우선순위 순)**

1. **Stereo Depth — LiDAR 없이 metric depth**
   > PSMNet-lite 직접 구현 (4D Cost Volume + Soft Argmin).
   > EPE **1.406px** (Classical 6.081 대비 4.3배↓), Depth MAE **2.797m** (카메라만으로 metric depth).

2. **3D MOT (AB3DMOT)**
   > 3D Kalman Filter(9D 상태벡터), Sutherland-Hodgman BEV 회전 IoU 직접 구현.
   > MOTA **0.8605** / MOTP **0.7222** / ID_SW = **0**

3. **포인트클라우드 (PointNet++, PointPillars)**
   > 두 논문 직접 구현 + KITTI 3D Detection 평가 파이프라인.

4. **Optical Flow**
   > RAFT flow + Semantic mask 결합 → detection 없이 움직이는 차량/보행자 탐지.

**자기소개 첫 문장 예시**
> "카메라만으로 metric depth를 추정하는 PSMNet-lite를 직접 구현하고, 3D Kalman Filter 기반 AB3DMOT로 MOTA 0.86을 달성했습니다."

---

### D. AI 플랫폼 / VLM (카카오, 네이버, LG AI Research, KT)

**JD 키워드**: VLM, 멀티모달, FastAPI, Docker, 마이크로서비스, QLoRA, 추론 서빙

**강조할 것 (우선순위 순)**

1. **VLM 파인튜닝 + 서빙 end-to-end**
   > Qwen2-VL-2B QLoRA(4bit) 파인튜닝, ROUGE-L **0.027 → 0.759** (+28배).
   > FastAPI + Uvicorn 추론 서버, Swagger UI 문서화.

2. **마이크로서비스 아키텍처**
   > Detection/Depth/Seg/VLM Gateway 5개 서비스 + React 프론트엔드.
   > `docker compose up --build` 단일 명령 실행, asyncio.gather 병렬 추론.

3. **기존 NLP/LLM 강점 강조**
   > RAG, PEFT, Transformer 파인튜닝 경험 → CV 도메인으로 확장.
   > ROUGE-L을 LCS DP로 직접 구현 (라이브러리 한국어 버그 우회).

**자기소개 첫 문장 예시**
> "Qwen2-VL-2B를 QLoRA로 파인튜닝해 Driving VQA ROUGE-L을 28배 향상시키고, FastAPI 마이크로서비스로 배포했습니다."

---

## 기술 면접 예상 질문 — 유형별

### Coding / 알고리즘 문제

면접에서 "직접 구현해보세요" 요청이 올 수 있는 것들:

| 구현 요청 가능성 | 준비 상태 |
|----------------|-----------|
| IoU 계산 (xyxy) | ✅ phase1, ab3dmot에서 구현 |
| Kalman Filter predict/update | ✅ ByteTrack, AB3DMOT에서 구현 |
| Hungarian Algorithm | ✅ ByteTrack, AB3DMOT, DETR에서 구현 |
| NMS (Non-Max Suppression) | ✅ YOLOv8 평가 파이프라인에서 이해 |
| mAP 계산 | ✅ phase1 평가 파이프라인에서 구현 |
| EPE / RMSE / delta1 | ✅ depth 파이프라인에서 구현 |
| Soft Argmin | ✅ PSMNet-lite에서 구현 |
| ROUGE-L (LCS) | ✅ VLM 평가에서 DP로 구현 |

---

## 포트폴리오 링크 준비

지원서에 첨부할 링크:

```
GitHub: https://github.com/[username]/autonomous_cv_pipeline
```

README에서 바로 확인 가능한 수치:
- mAP@50: 0.43 → 0.68
- RMSE: 5.44 → 2.71m
- MOTA: 0.9412 (ByteTrack)
- ROUGE-L: 0.027 → 0.759
- FPS: 132 → 316 (TensorRT)

---

## 면접 당일 체크리스트

**전날 밤**
- [ ] 수치 암기 카드 (INTERVIEW_PREP.md ## 빠른 수치 암기 카드) 10분 복습
- [ ] 지원 회사 유형에 맞는 "강조 3가지" 다시 확인

**면접 중**
- [ ] 수치 먼저 → 이유 설명 순서로
- [ ] "직접 구현했다"는 말 뒤에 항상 구체적인 컴포넌트 이름 붙이기
  - BAD: "ByteTrack 직접 구현했습니다"
  - GOOD: "Kalman Filter 상태 벡터(cx,cy,w,h,vx,vy), IoU 기반 헝가리안 매칭, tentative/confirmed/lost 상태 전이를 직접 구현했습니다"
- [ ] 부정적 결과 질문 나오면 — DR 실패, GradNorm 역효과 중 하나 준비해서 답변

---

## 연봉 협상 레버리지 포인트

- 논문 구현 수: 14개 이상 (단순 API 호출 아님)
- 평가 파이프라인 직접 구축 경험
- 부정적 결과 정량 분석 능력 (실무 필수)
- NLP/LLM 강점 보유 → VLM 트렌드에서 차별화
- FastAPI + Docker 배포 경험 (모델 서빙 end-to-end)
