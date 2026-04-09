# Dashboard TROUBLESHOOTING

에러 발생 시 여기에 누적 기록: 증상 / 원인 / 해결 / 교훈

---

## [2026-04-02] apt-get install 실패 — exit code 100

**증상**
```
RUN apt-get update && apt-get install -y libgl1-mesa-glx ...
exit code: 100
```

**원인**
`libgl1-mesa-glx`는 Debian bullseye까지의 구버전 패키지명.
`python:3.11-slim` 이미지가 Debian bookworm 기반으로 바뀌면서 해당 패키지가 사라짐.

**해결**
4개 Dockerfile(detection/depth/segmentation/vlm) 모두 `libgl1-mesa-glx` → `libgl1` 로 교체.

**교훈**
`python:3.11-slim`은 bookworm 기반. OpenCV용 libGL은 `libgl1`만 있으면 됨.

---

## [2026-04-02] Frontend 빌드 실패 — TS2339 import.meta.env

**증상**
```
src/api/client.ts(5,26): error TS2339: Property 'env' does not exist on type 'ImportMeta'.
exit code: 2
```

**원인**
`tsconfig.json`에 Vite 타입 선언(`vite/client`)이 없어서
TypeScript가 `import.meta.env`를 알 수 없음.

**해결**
`dashboard/frontend/tsconfig.json` compilerOptions에 추가:
```json
"types": ["vite/client"]
```

**교훈**
Vite + TypeScript 조합에서 `import.meta.env` 쓸 때는 반드시 `types: ["vite/client"]` 필요.
React + Vite 프로젝트 생성 시 기본 포함되지만, 직접 작성 시 빠지기 쉬움.

---

## [2026-04-02] VLM 컨테이너 unhealthy — C compiler not found

**증상**
```
RuntimeError: Failed to find C compiler. Please specify via CC environment variable.
dependency failed to start: container dashboard-vlm-1 is unhealthy
```

**원인**
`bitsandbytes` 4bit 양자화가 내부적으로 `triton`을 사용하고,
triton이 CUDA 유틸 빌드 시 C 컴파일러(`gcc`) 필요.
`python:3.11-slim`에는 gcc가 기본 포함되지 않음.

**해결**
`dashboard/services/vlm/Dockerfile` apt-get install에 `gcc` 추가:
```
libgl1 libglib2.0-0 gcc
```

**교훈**
`bitsandbytes` / `triton` 쓰는 서비스는 slim 이미지에 반드시 `gcc` 포함.
→ 그래도 `stdlib.h` 헤더 문제가 남을 수 있음 (아래 항목 참고).

---

## [2026-04-02] VLM 컨테이너 unhealthy — stdlib.h not found (triton 빌드 실패)

**증상**
```
fatal error: stdlib.h: No such file or directory
compilation terminated.
bitsandbytes import 실패 → Application startup failed
```

**원인**
`bitsandbytes`가 `triton`을 통해 CUDA 유틸을 런타임에 C 컴파일.
`gcc`는 있어도 C 표준 헤더(`libc6-dev`)가 slim 이미지에 없음.
근본적으로 Docker 컨테이너 안에서 bitsandbytes 4bit 양자화는 의존성 지옥.

**해결**
4bit 양자화 제거 → float16으로 대체:
- `main.py`: `BitsAndBytesConfig` 삭제, `torch_dtype=torch.float16` 사용
- `requirements.txt`: `bitsandbytes` 제거

**교훈**
Qwen2-VL-2B는 2B 모델 → float16으로도 ~4GB VRAM, RTX 4080 SUPER에서 충분.
Docker 컨테이너 안 bitsandbytes는 triton 빌드 의존성으로 불안정. 피하는 게 낫다.

---

## [2026-04-03] VLM LoRA 로드 실패 — peft 버전 불일치

**증상**
```
TypeError: LoraConfig.__init__() got an unexpected keyword argument 'alora_invocation_tokens'
```

**원인**
로컬 Anaconda 환경에서 `peft 0.18.1`로 LoRA 어댑터를 저장했으나
Docker requirements.txt에 `peft==0.13.2`가 지정되어 있어서 새 파라미터를 모름.

**해결**
1. `dashboard/services/vlm/requirements.txt`: `peft==0.13.2` → `peft==0.18.1` (로컬 버전과 동일하게 고정)
2. Docker 캐시가 이전 pip install을 재사용하므로 반드시 `--no-cache` 로 빌드:
   ```bash
   docker compose build --no-cache vlm && docker compose up vlm
   ```

**교훈**
- LoRA 어댑터 저장 환경(로컬)과 서빙 환경(Docker)의 PEFT 버전은 반드시 동일해야 함
- 로컬 버전 확인: `python -c "import peft; print(peft.__version__)"`
- requirements.txt 변경 후 `docker compose up --build`는 일부 레이어가 캐시되어 pip 업그레이드가 실제로 안 될 수 있음
- 패키지 버전 변경 시에는 항상 `--no-cache` 사용

---
