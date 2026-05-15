"""
pre_annotate.py
===============
Genera pre-anotaciones automaticas para las imagenes de PRUEBA5.

Estrategia hibrida:
  - rayadura (clase 0): modelo YOLOv8 entrenado (best.pt) a conf muy baja
                        para maximizar recall (se revisara manualmente)
  - agujero  (clase 1): OpenCV conservador — solo manchas oscuras compactas

Uso:
  python src/pre_annotate.py --sample        # 30 imagenes de revision visual
  python src/pre_annotate.py --all           # todo el dataset
  python src/pre_annotate.py --image foto.jpg
"""

import argparse
import json
import random
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ─── Rutas ───────────────────────────────────────────────────────────────────
_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_DIR  = _SCRIPT_DIR.parent
_ROOT_DIR     = _PROJECT_DIR.parent
_PRUEBA5_DIR  = _ROOT_DIR / "PRUEBAS" / "PRUEBA5" / "base_images"
_CSV_PATH     = _PRUEBA5_DIR / "_classes.csv"
_OUT_DIR      = _PROJECT_DIR / "data" / "pre_annotations"
_MODEL_PATH   = _PROJECT_DIR / "outputs" / "runs" / "scratch_yolo" / "weights" / "best.pt"

CLASSES       = {0: "rayadura", 1: "agujero"}
CLASS_COLORS  = {0: (0, 80, 220), 1: (0, 200, 50)}   # BGR

# ─── Parametros ───────────────────────────────────────────────────────────────
MODEL_CONF      = 0.05    # confianza muy baja → maximo recall (el usuario corregira FP)
MODEL_IOU       = 0.40
HOLE_MIN_AREA   = 0.00008 # fraccion minima de imagen
HOLE_MAX_AREA   = 0.10    # fraccion maxima de imagen
HOLE_DARK_THRES = 65      # umbral para regiones muy oscuras (solo lo realmente oscuro)
HOLE_AR_MAX     = 3.0     # relacion de aspecto maxima para agujero
HOLE_CIRC_MIN   = 0.12    # circularidad minima (mas permisivo para chips irregulares)

# Objetos OpenCV reutilizables (costosos de crear por imagen)
_CLAHE  = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_K_E5   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_K_E9   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_K_E7   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_K_E3   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# ─── Utilidades ──────────────────────────────────────────────────────────────

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / (ua + 1e-6)


def nms_merge(boxes, iou_thresh=0.3):
    if not boxes:
        return []
    boxes = list(set(boxes))
    kept  = []
    while boxes:
        ref = boxes.pop(0)
        group = [ref]
        remaining = []
        for b in boxes:
            if _iou(ref, b) > iou_thresh:
                group.append(b)
                ref = (min(g[0] for g in group), min(g[1] for g in group),
                       max(g[2] for g in group), max(g[3] for g in group))
            else:
                remaining.append(b)
        kept.append(ref)
        boxes = remaining
    return kept


def tile_mask(gray):
    """
    Para imagenes de una sola baldosa sobre fondo negro:
    devuelve mascara booleana con el interior de la baldosa.
    Si no hay fondo negro claro, devuelve mascara completa.
    """
    dark_pct = np.mean(gray < 30)
    if dark_pct < 0.05:          # no hay fondo negro → imagen de suelo
        return np.ones(gray.shape, dtype=bool)
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask.astype(bool)


# ─── Deteccion de agujeros (OpenCV) ──────────────────────────────────────────

def detect_agujeros(img_bgr):
    """
    Detecta imperfecciones redondas/compactas de cualquier tipo:
      - Manchas oscuras en superficie (pits, holes, dark spots)
      - Chips en bordes (color marron/naranja - arcilla expuesta)
      - Manchas de color diferente al fondo
    Devuelve lista de (x1,y1,x2,y2) en pixeles.
    """
    H, W     = img_bgr.shape[:2]
    min_area = H * W * HOLE_MIN_AREA
    max_area = H * W * HOLE_MAX_AREA

    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    enh   = _CLAHE.apply(gray)
    tmask = tile_mask(gray)

    all_boxes = []

    # ── 1. Manchas oscuras compactas (pits, agujeros oscuros) ────────────────
    _, dark = cv2.threshold(enh, HOLE_DARK_THRES, 255, cv2.THRESH_BINARY_INV)
    dark[~tmask] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,  _K_E5)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, _K_E9)

    for cnt in cv2.findContours(dark, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar   = max(w, h) / (min(w, h) + 1e-5)
        peri = cv2.arcLength(cnt, True)
        circ = 4 * np.pi * area / (peri ** 2 + 1e-5)
        if ar <= HOLE_AR_MAX and circ >= HOLE_CIRC_MIN:
            all_boxes.append((x, y, x+w, y+h))

    # ── 2. Chips de borde (color marron/naranja - arcilla expuesta) ──────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Marron claro / naranja (chips tipicos de ceramica)
    m1 = cv2.inRange(hsv, np.array([5,  40,  20]), np.array([35, 255, 180]))
    # Marron oscuro (chips en sombra)
    m2 = cv2.inRange(hsv, np.array([0,  25,  10]), np.array([20, 200, 100]))
    # Marron rojizo
    m3 = cv2.inRange(hsv, np.array([160, 30, 20]), np.array([180, 200, 150]))

    brown = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)
    brown[~tmask] = 0   # excluir fondo negro

    brown = cv2.morphologyEx(brown, cv2.MORPH_CLOSE, _K_E7)
    brown = cv2.morphologyEx(brown, cv2.MORPH_OPEN,  _K_E3)

    for cnt in cv2.findContours(brown, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(cnt)
        if area < min_area * 0.3 or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        ar = max(w, h) / (min(w, h) + 1e-5)
        if ar <= HOLE_AR_MAX * 1.5:   # chips pueden ser algo alargados
            all_boxes.append((x, y, x+w, y+h))

    return nms_merge(all_boxes, iou_thresh=0.25)


# ─── Deteccion de rayaduras (modelo YOLO) ────────────────────────────────────

_model_cache = {}

def get_model():
    if "model" not in _model_cache:
        from ultralytics import YOLO
        print(f"[MODEL] Cargando {_MODEL_PATH.name} ...")
        _model_cache["model"] = YOLO(str(_MODEL_PATH))
    return _model_cache["model"]


def detect_rayaduras_model(img_path: Path):
    """
    Usa best.pt a conf muy baja para maximizar recall.
    Devuelve lista de (x1,y1,x2,y2).
    """
    if not _MODEL_PATH.exists():
        return []
    model   = get_model()
    results = model.predict(
        source  = str(img_path),
        conf    = MODEL_CONF,
        iou     = MODEL_IOU,
        imgsz   = 640,
        verbose = False,
    )
    boxes = []
    for r in results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            boxes.append((x1, y1, x2, y2))
    return boxes


# ─── Pipeline por imagen ──────────────────────────────────────────────────────

def process_image(img_path: Path):
    """
    Devuelve lista de (cls_id, cx, cy, bw, bh) normalizados [0-1].
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    H, W = img.shape[:2]

    rayaduras_px = detect_rayaduras_model(img_path)
    agujeros_px  = detect_agujeros(img)

    def px_to_yolo(boxes, cls_id):
        out = []
        for x1, y1, x2, y2 in boxes:
            cx = np.clip((x1 + x2) / 2 / W, 0.001, 0.999)
            cy = np.clip((y1 + y2) / 2 / H, 0.001, 0.999)
            bw = np.clip((x2 - x1) / W,     0.001, 1.0)
            bh = np.clip((y2 - y1) / H,     0.001, 1.0)
            out.append((cls_id, float(cx), float(cy), float(bw), float(bh)))
        return out

    return px_to_yolo(rayaduras_px, 0) + px_to_yolo(agujeros_px, 1)


# ─── Guardar label YOLO ───────────────────────────────────────────────────────

def save_yolo_label(out_path: Path, labels: list):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for cls_id, cx, cy, bw, bh in labels:
            f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


# ─── Visualizacion ────────────────────────────────────────────────────────────

def draw_annotations(img_bgr: np.ndarray, labels: list) -> np.ndarray:
    H, W = img_bgr.shape[:2]
    out  = img_bgr.copy()
    for cls_id, cx, cy, bw, bh in labels:
        x1 = int((cx - bw/2) * W)
        y1 = int((cy - bh/2) * H)
        x2 = int((cx + bw/2) * W)
        y2 = int((cy + bh/2) * H)
        color = CLASS_COLORS.get(int(cls_id), (200, 200, 200))
        thick = 3 if cls_id == 0 else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = CLASSES[int(cls_id)]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ─── Formato Label Studio ─────────────────────────────────────────────────────

def to_ls_task(img_path: Path, labels: list, doc_root: Path) -> dict:
    img  = cv2.imread(str(img_path))
    H, W = (img.shape[:2] if img is not None else (640, 640))
    try:
        rel = img_path.relative_to(doc_root).as_posix()
    except ValueError:
        rel = img_path.as_posix()

    task = {
        "data": {"image": f"/data/local-files/?d={rel}"},
        "predictions": [{
            "model_version": "yolo-preannotation-v1",
            "score": 0.7,
            "result": []
        }]
    }
    for cls_id, cx, cy, bw, bh in labels:
        task["predictions"][0]["result"].append({
            "id":           str(uuid.uuid4())[:8],
            "type":         "rectanglelabels",
            "from_name":    "label",
            "to_name":      "image",
            "original_width":  W,
            "original_height": H,
            "value": {
                "x":      round((cx - bw/2) * 100, 3),
                "y":      round((cy - bh/2) * 100, 3),
                "width":  round(bw * 100, 3),
                "height": round(bh * 100, 3),
                "rotation": 0,
                "rectanglelabels": [CLASSES[int(cls_id)]]
            }
        })
    return task


# ─── Lectura CSV ──────────────────────────────────────────────────────────────

def read_csv(csv_path: Path) -> dict:
    import csv as _csv
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            result[row["filename"]] = {k: int(v) for k, v in row.items()
                                       if k != "filename"}
    return result


# ─── Ejecucion principal ──────────────────────────────────────────────────────

def run(image_paths: list):
    label_dir  = _OUT_DIR / "labels"
    sample_dir = _OUT_DIR / "samples"
    ls_json    = _OUT_DIR / "label_studio_tasks.json"

    for d in [_OUT_DIR, label_dir, sample_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tasks  = []
    counts = defaultdict(int)
    no_det = 0

    print(f"\n[PRE-ANNOTATE] {len(image_paths)} imagenes")
    print(f"  Modelo: {_MODEL_PATH.name if _MODEL_PATH.exists() else 'NO ENCONTRADO'}")
    print(f"  conf={MODEL_CONF}  iou={MODEL_IOU}\n")

    doc_root = _ROOT_DIR

    for i, img_path in enumerate(image_paths, 1):
        labels = process_image(img_path)

        save_yolo_label(label_dir / (img_path.stem + ".txt"), labels)

        img = cv2.imread(str(img_path))
        if img is not None:
            ann = draw_annotations(img, labels)
            h, w = ann.shape[:2]
            if max(h, w) > 900:
                s   = 900 / max(h, w)
                ann = cv2.resize(ann, None, fx=s, fy=s)
            cv2.imwrite(str(sample_dir / img_path.name), ann)

        tasks.append(to_ls_task(img_path, labels, doc_root))

        for cls_id, *_ in labels:
            counts[CLASSES[int(cls_id)]] += 1
        if not labels:
            no_det += 1

        if i % 10 == 0 or i == len(image_paths):
            print(f"  {i:4d}/{len(image_paths)}  "
                  f"rayaduras={counts['rayadura']}  "
                  f"agujeros={counts['agujero']}  "
                  f"sin_det={no_det}")

    with open(ls_json, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*55}")
    print("  RESUMEN")
    print(f"{'='*55}")
    print(f"  Imagenes procesadas:   {len(image_paths)}")
    print(f"  Sin deteccion:         {no_det}")
    for name, cnt in counts.items():
        print(f"  {name+':':25s} {cnt}")
    print(f"  Total anotaciones:     {sum(counts.values())}")
    print(f"  Label Studio JSON:     {ls_json}")
    print(f"  Muestras visuales:     {sample_dir}")
    print(f"{'='*55}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--all",    action="store_true")
    ap.add_argument("--image",  type=str, default=None)
    ap.add_argument("--n",      type=int, default=30)
    args = ap.parse_args()

    exts     = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    all_imgs = sorted(p for p in _PRUEBA5_DIR.iterdir()
                      if p.suffix.lower() in exts)

    if args.image:
        paths = [Path(args.image)]

    elif args.sample:
        csv_data = read_csv(_CSV_PATH)
        by_type  = defaultdict(list)
        for fname, lbl in csv_data.items():
            p = _PRUEBA5_DIR / fname
            if not p.exists():
                continue
            if lbl.get("crack"):     by_type["crack"].append(p)
            elif lbl.get("scratch"): by_type["scratch"].append(p)
            elif lbl.get("stain"):   by_type["stain"].append(p)
            else:                    by_type["normal"].append(p)

        random.seed(42)
        n_each = max(1, args.n // 4)
        paths  = []
        for cat, imgs in by_type.items():
            sample = random.sample(imgs, min(n_each, len(imgs)))
            print(f"  [{cat}] {len(sample)} imagenes")
            paths += sample
        paths = paths[:args.n]

    elif args.all:
        paths = all_imgs

    else:
        print("Usa --sample, --all o --image.")
        return

    run(paths)


if __name__ == "__main__":
    main()
