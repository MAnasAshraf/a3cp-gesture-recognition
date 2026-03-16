import requests
import time

try:
    requests.post("http://localhost:8000/api/inference/start")
    print("Started inference.")
except Exception as e:
    print("Failed to start:", e)

for i in range(10):
    try:
        r = requests.get("http://localhost:8000/api/inference/prediction")
        print(f"Prediction {i}: {r.status_code}")
    except Exception as e:
        print(f"Error {i}: {e}")
    time.sleep(0.5)

print("Done.")
