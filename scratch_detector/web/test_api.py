"""Quick smoke-test for the /analyze endpoint."""
import json, urllib3

IMG = r"C:\Users\javie\PycharmProjects\VISION_ARTIFICIAL\testing\pruebas_detector\PRUEBA5\base_images\_MG_2292_jpg.rf.V0bBkDEkvGS7Zgq5yfrY.jpg"

with open(IMG, "rb") as f:
    img_data = f.read()

http = urllib3.PoolManager()
resp = http.request(
    "POST",
    "http://localhost:8000/analyze",
    fields={
        "file": ("test.jpg", img_data, "image/jpeg"),
        "tile_cm": "60",
        "conf": "0.25",
    },
)
data = json.loads(resp.data)
print("Status:", resp.status)

if "result_id" in data:
    print("result_id:", data["result_id"])
    r2  = http.request("GET", f"http://localhost:8000/results/{data['result_id']}")
    res = json.loads(r2.data)
    print("verdict: ", res["verdict"])
    print("defects: ", len(res["defects"]))
    print("elapsed: ", res["elapsed_ms"], "ms")
    print("unit:    ", res["unit"])
    for d in res["defects"][:4]:
        print(f"  [{d['type_en']}] {d['conf']:.0%}  {d['size_str']}")
    print(f"\n  Dashboard URL: http://localhost:8000/static/dashboard.html?id={data['result_id']}")
else:
    print("Error:", data)
