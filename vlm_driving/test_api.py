import requests
import json

IMG = r"C:\Users\apple\Desktop\autonomous_cv_pipeline\phase4_carla\data_collection\carla_dataset\images\000010.jpg"
BASE = "http://127.0.0.1:8000"

# 헬스체크
r = requests.get(f"{BASE}/health")
print("=== health ===")
print(r.json())
print()

# 7종 배치
with open(IMG, "rb") as f:
    r = requests.post(f"{BASE}/predict/batch", files={"file": f})
data = r.json()
print("=== predict/batch ===")
print(f"응답시간: {data['elapsed_sec']}s")
print()
for k, v in data["results"].items():
    print(f"[{k}]")
    print(f"  {v}")
    print()
