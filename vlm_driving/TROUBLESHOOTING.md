# VLM 프로젝트 — 문제 해결 로그

---

## [2026-03-31] peft ImportError: HybridCache

**증상**
```
ImportError: cannot import name 'HybridCache' from 'transformers'
```

**원인**
- trl 설치 시 transformers가 4.56.x → 5.4.0으로 자동 업그레이드
- transformers 5.x에서 `HybridCache`가 제거/이름변경
- 기존 peft 0.17.1은 `HybridCache` import를 시도 → 충돌

**해결**
```bash
pip install "peft==0.18.1"
```
peft 0.18.1이 transformers 5.x 호환

**최종 동작 버전**
- transformers 5.4.0
- peft 0.18.1
- trl 1.0.0
- qwen_vl_utils (최신)

**교훈**
- trl/peft 설치 시 항상 transformers 버전 확인 후 peft 버전 맞춰서 설치
- `pip install trl "peft>=0.18.0" qwen-vl-utils -q` 한번에 설치 권장

---
