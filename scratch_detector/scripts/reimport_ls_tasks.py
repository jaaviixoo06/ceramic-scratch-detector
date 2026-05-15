"""
Regenera el JSON de Label Studio usando URLs http://localhost:8082/
y lo importa al proyecto 11.
"""
import json, os, requests, uuid
from pathlib import Path

BASE        = "http://localhost:8080"
IMG_SERVER  = "http://localhost:8082"
PROJECT_ID  = 11
REFRESH     = os.environ.get("LS_REFRESH_TOKEN") or input("Pega el refresh token de Label Studio: ").strip()

_SCRIPT_DIR = Path(__file__).resolve().parent
TASKS_JSON  = _SCRIPT_DIR.parent / "data" / "pre_annotations" / "label_studio_tasks.json"
CLASSES     = {0: "rayadura", 1: "agujero"}

# Auth
access = requests.post(f"{BASE}/api/token/refresh",
                       json={"refresh": REFRESH}).json()["access"]
H = {"Authorization": f"Bearer {access}", "Content-Type": "application/json"}
print("Auth OK")

# Load original tasks and rebuild with http URLs
with open(TASKS_JSON, encoding="utf-8") as f:
    original = json.load(f)

new_tasks = []
for t in original:
    # Extract filename from the ?d= param
    old_url = t["data"]["image"]               # /data/local-files/?d=PRUEBAS/...
    rel     = old_url.split("?d=")[-1]         # PRUEBAS/PRUEBA5/base_images/img.jpg
    fname   = rel.split("/")[-1]               # img.jpg
    new_url = f"{IMG_SERVER}/{fname}"

    new_task = {"data": {"image": new_url}, "predictions": t.get("predictions", [])}
    new_tasks.append(new_task)

print(f"Reimportando {len(new_tasks)} tareas con URLs http://localhost:8082/ ...")

BATCH = 100
imported = 0
for i in range(0, len(new_tasks), BATCH):
    batch = new_tasks[i:i+BATCH]
    r = requests.post(f"{BASE}/api/projects/{PROJECT_ID}/import",
                      headers=H, json=batch)
    if r.status_code in (200, 201):
        imported += len(batch)
    else:
        print(f"ERROR batch {i}: {r.status_code} {r.text[:120]}")
        break
    if imported % 500 == 0 or imported >= len(new_tasks):
        print(f"  {imported}/{len(new_tasks)}")

print(f"\nListo. Abre: {BASE}/projects/{PROJECT_ID}/")
