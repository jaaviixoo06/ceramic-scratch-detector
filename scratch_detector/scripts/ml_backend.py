"""
ml_backend.py
=============
Servidor ML Backend para Label Studio.
Usa el modelo YOLOv8s-seg entrenado para pre-anotar automaticamente.

Uso:
  python scripts/ml_backend.py          # arranca en puerto 9090
  python scripts/ml_backend.py --port 9091

En Label Studio:
  Settings -> Model -> Add Model
  URL: http://localhost:9090
  Activar "Use for interactive preannotations"
"""

import argparse
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("ERROR: pip install flask")
    sys.exit(1)

# ── Rutas ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_MODEL_PATH  = _PROJECT_DIR / "outputs" / "runs" / "scratch_yolo" / "weights" / "best.pt"

# ── Clase/config — debe coincidir con el label_config del proyecto ─────────────
# Rayadura → PolygonLabels  (from_name="polygon_label")
# Agujero  → RectangleLabels (from_name="label")
RAYADURA_CLS = 1
AGUJERO_CLS  = 0

app   = Flask(__name__)
model = None


def get_model():
    global model
    if model is None:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: pip install ultralytics")
            sys.exit(1)
        print(f"[MODEL] Cargando {_MODEL_PATH.name} ...")
        model = YOLO(str(_MODEL_PATH))
    return model


def load_image_from_url(url: str) -> np.ndarray:
    """Descarga imagen desde URL y devuelve array BGR."""
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = resp.read()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def yolo_to_ls_predictions(result, conf_thresh: float = 0.20) -> list:
    """
    Convierte resultado YOLO al formato de predicciones de Label Studio.
    - Rayadura → polygonlabels  (usa mascara de segmentacion)
    - Agujero  → rectanglelabels
    """
    annotations = []
    img_h, img_w = result.orig_shape
    has_masks = result.masks is not None and len(result.masks) > 0

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if conf < conf_thresh:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if cls_id == RAYADURA_CLS and has_masks:
            # Poligono de segmentacion para rayadura
            poly = result.masks.xy[i]
            if len(poly) < 3:
                continue
            points = [[round(float(px) / img_w * 100, 2),
                       round(float(py) / img_h * 100, 2)]
                      for px, py in poly]
            annotations.append({
                "id":              f"r_{i}",
                "type":            "polygonlabels",
                "from_name":       "polygon_label",
                "to_name":         "image",
                "original_width":  img_w,
                "original_height": img_h,
                "value": {
                    "points":        points,
                    "polygonlabels": ["rayadura"],
                    "closed":        True,
                },
                "score": conf,
            })
        else:
            # Rectangulo para agujero (o rayadura sin mascara)
            label     = "agujero" if cls_id == AGUJERO_CLS else "rayadura"
            from_name = "label"
            annotations.append({
                "id":              f"r_{i}",
                "type":            "rectanglelabels",
                "from_name":       from_name,
                "to_name":         "image",
                "original_width":  img_w,
                "original_height": img_h,
                "value": {
                    "x":               round(x1 / img_w * 100, 2),
                    "y":               round(y1 / img_h * 100, 2),
                    "width":           round((x2 - x1) / img_w * 100, 2),
                    "height":          round((y2 - y1) / img_h * 100, 2),
                    "rotation":        0,
                    "rectanglelabels": [label],
                },
                "score": conf,
            })

    return annotations


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "UP", "model": _MODEL_PATH.name})


@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({"model_version": _MODEL_PATH.stem})


@app.route("/predict", methods=["POST"])
def predict():
    data  = request.json or {}
    tasks = data.get("tasks", [])

    m       = get_model()
    results = []

    for task in tasks:
        image_url = task.get("data", {}).get("image", "")
        if not image_url:
            results.append({"result": [], "score": 0})
            continue

        try:
            img = load_image_from_url(image_url)
            if img is None:
                raise ValueError("Imagen no cargada")
            yolo_results = m.predict(
                source=img, conf=0.20, iou=0.45, imgsz=640, verbose=False
            )
            annotations = yolo_to_ls_predictions(yolo_results[0])
            avg_score   = (sum(a["score"] for a in annotations) / len(annotations)
                           if annotations else 0.0)
            results.append({"result": annotations, "score": round(avg_score, 3)})
            print(f"  [{task.get('id')}] {len(annotations)} predicciones")
        except Exception as e:
            print(f"  ERROR en tarea {task.get('id')}: {e}")
            results.append({"result": [], "score": 0})

    return jsonify({"results": results})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9090)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()

    get_model()   # precarga el modelo al arrancar
    print(f"\n ML Backend listo en http://localhost:{args.port}")
    print("  En Label Studio: Settings -> Model -> Add Model -> http://localhost:9090\n")
    app.run(host=args.host, port=args.port, debug=False)
