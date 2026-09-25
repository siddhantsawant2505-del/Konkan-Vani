import requests
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("=== 1. GET /health ===")
r1 = requests.get("http://127.0.0.1:8000/health")
print("Status:", r1.status_code)
print("Response:", json.dumps(r1.json(), indent=2))

print("\n=== 2. POST /api/resolve (Known Idiom: 'Haat dakhvun ayaak') ===")
r2 = requests.post("http://127.0.0.1:8000/api/resolve", json={"text": "Haat dakhvun ayaak"})
print("Status:", r2.status_code)
print("Response:\n", json.dumps(r2.json(), indent=2, ensure_ascii=False))

print("\n=== 3. POST /api/resolve (Misspelled Variant: 'haat dakhwun aayaak') ===")
r3 = requests.post("http://127.0.0.1:8000/api/resolve", json={"text": "haat dakhwun aayaak"})
print("Status:", r3.status_code)
print("Response:\n", json.dumps(r3.json(), indent=2, ensure_ascii=False))

print("\n=== 4. PHASE 4: Unseen Idiom-like Sentence ('Dori bandun udkaant marap uddi') ===")
r4 = requests.post("http://127.0.0.1:8000/api/resolve", json={"text": "Dori bandun udkaant marap uddi"})
print("Status:", r4.status_code)
print("Response:\n", json.dumps(r4.json(), indent=2, ensure_ascii=False))
