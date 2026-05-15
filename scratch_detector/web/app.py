"""
web/app.py
==========
FastAPI backend — VISION_TILE_CORE.

Ejecutar desde scratch_detector/:
    uvicorn web.app:app --reload --port 8000

Instalar dependencias si faltan:
    pip install fastapi uvicorn python-multipart
"""

import base64
import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

_WEB_DIR     = Path(__file__).resolve().parent
_PROJECT_DIR = _WEB_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR / "src"))

from infer import CLASS_CONFIG, detect_tile_px, draw_detections  # noqa: E402

app = FastAPI(title="VISION_TILE_CORE")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount(
    "/static",
    StaticFiles(directory=str(_WEB_DIR / "static"), html=True),
    name="static",
)

# ── Model singleton ────────────────────────────────────────────────────────────

_model = None


def _find_model():
    weights = _PROJECT_DIR / "outputs" / "runs" / "scratch_yolo" / "weights"
    for name in ("best5.pt", "best4.pt", "best.pt", "last.pt"):
        p = weights / name
        if p.exists():
            return p
    return None


def get_model():
    global _model
    if _model is None:
        path = _find_model()
        if path is None:
            raise RuntimeError(
                "No se encontró ningún modelo en outputs/runs/scratch_yolo/weights/"
            )
        from ultralytics import YOLO
        _model = YOLO(str(path))
        print(f"[MODEL] Cargado: {path.name}")
    return _model


# ── In-memory results cache (últimos 50 análisis) ──────────────────────────────

_cache: OrderedDict = OrderedDict()


def cache_put(rid: str, data: dict):
    _cache[rid] = data
    while len(_cache) > 50:
        _cache.popitem(last=False)


# ── Helpers ────────────────────────────────────────────────────────────────────

def to_b64(img: np.ndarray, q: int = 88) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()


def fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    s = max_side / max(h, w)
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return RedirectResponse("/static/index.html")


@app.post("/analyze")
async def analyze(
    file:    UploadFile = File(...),
    tile_cm: float      = Form(None),
    conf:    float      = Form(0.25),
):
    raw = await file.read()
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Imagen inválida o formato no soportado"}, status_code=400)

    H, W = img.shape[:2]
    scale = tile_info = None

    if tile_cm:
        det     = detect_tile_px(img)
        tile_mm = tile_cm * 10
        if det:
            tile_px = det[0]
            scale   = tile_mm / tile_px
            tile_info = {
                "tile_cm": tile_cm,
                "tile_px": round(tile_px),
                "scale":   round(scale, 4),
                "fallback": False,
            }
        else:
            tile_px = min(H, W)
            scale   = tile_mm / tile_px
            tile_info = {
                "tile_cm": tile_cm,
                "tile_px": tile_px,
                "scale":   round(scale, 4),
                "fallback": True,
            }

    t0      = time.perf_counter()
    results = get_model().predict(source=img, conf=conf, iou=0.45, imgsz=640, verbose=False)
    ms      = round((time.perf_counter() - t0) * 1000, 1)

    r                    = results[0]
    annotated, summary, measures = draw_detections(img, r, scale)
    unit = "mm" if scale else "px"

    defects = []
    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        conf_v = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        pad  = 20
        crop = img[max(0, y1 - pad):min(H, y2 + pad),
                   max(0, x1 - pad):min(W, x2 + pad)]

        m = measures[i] if i < len(measures) else {}

        if cls_id == 1:  # Rayadura
            l  = m.get("length_mm", m.get("length_px", 0))
            g  = m.get("width_mm",  m.get("width_px",  0))
            sz = f"L: {l:.0f} {unit}   G: {g:.1f} {unit}"
        else:            # Agujero
            lv = m.get("length_mm", m.get("length_px", 0))
            wv = m.get("width_mm",  m.get("width_px",  0))
            av = m.get("area_mm2",  m.get("area_px",   0))
            u2 = "mm²" if scale else "px²"
            sz = f"{lv:.0f}×{wv:.0f} {unit}   A: {av:.0f} {u2}"

        defects.append({
            "id":      i + 1,
            "cls_id":  cls_id,
            "type":    CLASS_CONFIG.get(cls_id, {}).get("name", f"cls_{cls_id}"),
            "type_en": "SCRATCH" if cls_id == 1 else "HOLE",
            "conf":    round(conf_v, 3),
            "size_str": sz,
            "measures": m,
            "crop_b64": to_b64(fit(crop, 256), q=82),
        })

    payload = {
        "annotated_b64": to_b64(fit(annotated, 1280), q=90),
        "summary":       {k: int(v) for k, v in summary.items()},
        "defects":       defects,
        "verdict":       "PASSED" if not defects else "FAILED",
        "elapsed_ms":    ms,
        "tile_info":     tile_info,
        "unit":          unit,
        "filename":      file.filename or "uploaded_image",
        "img_w":         W,
        "img_h":         H,
        "conf_thresh":   conf,
    }

    rid = str(uuid.uuid4())
    cache_put(rid, payload)
    return JSONResponse({"result_id": rid})


@app.get("/results/{rid}")
async def get_results(rid: str):
    if rid not in _cache:
        return JSONResponse({"error": "Resultado no encontrado o expirado"}, status_code=404)
    return JSONResponse(_cache[rid])
