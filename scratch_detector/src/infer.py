"""
infer.py
========
Inferencia con el modelo entrenado de rayaduras en baldosas.
Detecta rayaduras y agujeros, y calcula sus dimensiones.

Uso:
  python src/infer.py --image foto_baldosa.jpg
  python src/infer.py --image foto.jpg --tile-cm 60   # baldosa 60x60cm → mm reales
  python src/infer.py --image carpeta/ --no-show
  python src/infer.py --image foto.jpg --model outputs/runs/scratch_yolo/weights/best5.pt
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR    = Path(__file__).resolve().parent
_PROJECT_DIR   = _SCRIPT_DIR.parent
_DEFAULT_MODEL = _PROJECT_DIR / "outputs" / "runs" / "scratch_yolo" / "weights" / "best.pt"
_OUTPUT_DIR    = _PROJECT_DIR / "outputs" / "predictions"

CLASS_CONFIG = {
    0: {"name": "Agujero",  "color": (0, 200,  50), "thick": 2},
    1: {"name": "Rayadura", "color": (0,  80, 220), "thick": 2},
}

_CLAHE_BH = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
_K_CLEAN  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


# ── Medicion ───────────────────────────────────────────────────────────────────

def measure_defect(poly: np.ndarray | None, bbox: tuple, cls_id: int):
    """
    Calcula dimensiones del defecto a partir del poligono de mascara o bbox.
    Rayadura: longitud (eje mayor) y grosor (eje menor) via minAreaRect.
    Agujero:  ancho, alto y area del contorno.
    Devuelve dict con claves: length, width, area, angle (pixeles).
    """
    if poly is not None and len(poly) >= 5:
        cnt = poly.reshape(-1, 1, 2).astype(np.float32)
        _, (w, h), angle = cv2.minAreaRect(cnt)
        length = float(max(w, h))
        width  = float(min(w, h))
        area   = float(cv2.contourArea(cnt))
    else:
        x1, y1, x2, y2 = bbox
        length = float(max(x2 - x1, y2 - y1))
        width  = float(min(x2 - x1, y2 - y1))
        area   = float((x2 - x1) * (y2 - y1))
        angle  = 0.0

    return {"length": length, "width": width, "area": area}


def px_to_mm(px: float, scale: float) -> float:
    return px * scale


def format_measure(m: dict, cls_id: int, scale: float | None) -> str:
    """Devuelve cadena de medidas legible para superponer en imagen."""
    if cls_id == 1:  # Rayadura
        if scale:
            return (f"L:{m['length']*scale:.0f}mm "
                    f"G:{m['width']*scale:.1f}mm")
        return f"L:{m['length']:.0f}px  G:{m['width']:.1f}px"
    else:            # Agujero
        if scale:
            return (f"{m['length']*scale:.0f}x{m['width']*scale:.0f}mm "
                    f"A:{m['area']*scale*scale:.0f}mm²")
        return (f"{m['length']:.0f}x{m['width']:.0f}px "
                f"A:{m['area']:.0f}px²")


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Inferencia - detector de rayaduras en baldosas")
    p.add_argument("--image",    type=str,   required=True)
    p.add_argument("--model",    type=str,   default=None)
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou",      type=float, default=0.45)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--device",   type=str,   default=None)
    p.add_argument("--no-show",  action="store_true")
    p.add_argument("--save",     action="store_true", default=True)
    p.add_argument("--tile-cm",  type=float, default=None,
                   help="Tamaño real de la baldosa en cm (ej: 60). "
                        "Activa conversion de pixeles a mm.")
    return p.parse_args()


# ── Modelo ─────────────────────────────────────────────────────────────────────

def find_model(model_arg):
    if model_arg:
        p = Path(model_arg)
        if p.exists(): return p
        alt = _PROJECT_DIR / model_arg
        if alt.exists(): return alt
        print(f"ERROR: Modelo no encontrado: {model_arg}"); sys.exit(1)
    for candidate in [
        _DEFAULT_MODEL,
        _PROJECT_DIR / "outputs" / "runs" / "scratch_yolo" / "weights" / "last.pt",
    ]:
        if candidate.exists():
            print(f"[MODEL] Usando: {candidate}")
            return candidate
    print("ERROR: No se encontro ningun modelo."); sys.exit(1)


def collect_images(source):
    src = Path(source)
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if src.is_file():
        return [src] if src.suffix.lower() in exts else []
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
    print(f"ERROR: {source} no existe."); sys.exit(1)


# ── Blackhat ───────────────────────────────────────────────────────────────────

def extract_scratch_mask(image_bgr, x1, y1, x2, y2):
    H, W = image_bgr.shape[:2]
    pad = 6
    y1p, x1p = max(0, y1-pad), max(0, x1-pad)
    y2p, x2p = min(H, y2+pad), min(W, x2+pad)
    roi = image_bgr[y1p:y2p, x1p:x2p]
    if roi.size == 0:
        return np.zeros((H, W), dtype=np.uint8)
    roi_h, roi_w = roi.shape[:2]
    roi_area = roi_h * roi_w
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    enh  = _CLAHE_BH.apply(gray)
    bh_results = []
    for ks in (15, 25, 35):
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
        bh = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, k)
        bh_results.append(bh)
    blackhat = np.max(np.stack(bh_results, axis=0), axis=0)
    thresh = np.percentile(blackhat, 97)
    if thresh < 4:
        return np.zeros((H, W), dtype=np.uint8)
    mask = (blackhat >= thresh).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K_CLEAN)
    if np.count_nonzero(mask) > roi_area * 0.30:
        return np.zeros((H, W), dtype=np.uint8)
    k_join   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_big = cv2.dilate(mask, k_join)
    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(mask)
    min_area = roi_area * 0.0005
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        if len(cnt) >= 5:
            _, (ma, MA), _ = cv2.fitEllipse(cnt)
            ar = MA / (ma + 1e-5)
        else:
            bx, by, bw, bh2 = cv2.boundingRect(cnt)
            ar = max(bw, bh2) / (min(bw, bh2) + 1e-5)
        if ar > 2.0:
            cv2.drawContours(filtered, [cnt], -1, 255, -1)
    result = cv2.bitwise_and(mask, filtered)
    full = np.zeros((H, W), dtype=np.uint8)
    full[y1p:y2p, x1p:x2p] = result
    return full


# ── Dibujo + medicion ──────────────────────────────────────────────────────────

def draw_detections(image, detections, scale=None):
    """
    Dibuja mascaras, etiquetas y medidas.
    Devuelve (imagen_anotada, lista_de_medidas).
    Cada medida: dict con cls, name, conf, length, width, area (en px y mm si scale).
    """
    from collections import Counter
    annotated  = image.copy()
    summary    = Counter()
    measures   = []

    if detections is None or len(detections.boxes) == 0:
        return annotated, summary, measures

    has_masks = (detections.masks is not None and len(detections.masks) > 0)

    # Fill semitransparente de todas las mascaras en un solo pass
    if has_masks:
        fill_layer = annotated.copy()
        for i, _ in enumerate(detections.boxes):
            poly = detections.masks.xy[i].astype(np.int32)
            if len(poly) >= 3:
                cls_id = int(detections.boxes[i].cls[0])
                color  = CLASS_CONFIG.get(cls_id, {"color": (200,200,200)})["color"]
                cv2.fillPoly(fill_layer, [poly], color)
        cv2.addWeighted(fill_layer, 0.35, annotated, 0.65, 0, annotated)

    for i, box in enumerate(detections.boxes):
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        cfg    = CLASS_CONFIG.get(cls_id, {"name": f"cls_{cls_id}",
                                           "color": (200,200,200), "thick": 2})
        color  = cfg["color"]
        thick  = cfg["thick"]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        poly_pts = None
        if has_masks:
            poly_pts = detections.masks.xy[i].astype(np.int32)
            if len(poly_pts) >= 3:
                cv2.polylines(annotated, [poly_pts], isClosed=True,
                              color=color, thickness=thick)
            else:
                poly_pts = None
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thick)
        elif cls_id == 1:
            scratch_mask = extract_scratch_mask(image, x1, y1, x2, y2)
            if scratch_mask.any():
                overlay = annotated.copy()
                overlay[scratch_mask > 0] = color
                cv2.addWeighted(overlay, 0.45, annotated, 0.55, 0, annotated)
                contours, _ = cv2.findContours(scratch_mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(annotated, contours, -1, color, thick)
                if contours:
                    poly_pts = max(contours, key=cv2.contourArea).reshape(-1, 2)
            else:
                cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thick)
        else:
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, thick)

        # Medir
        m = measure_defect(poly_pts, (x1, y1, x2, y2), cls_id)

        # Etiqueta: nombre + conf + medidas
        conf_str    = f"{conf:.0%}"
        measure_str = format_measure(m, cls_id, scale)
        label_top   = f"{cfg['name']}  {conf_str}"
        label_bot   = measure_str

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        pad        = 3

        (tw1, th1), _ = cv2.getTextSize(label_top, font, font_scale, 1)
        (tw2, th2), _ = cv2.getTextSize(label_bot, font, font_scale, 1)
        box_w = max(tw1, tw2) + 2 * pad
        box_h = th1 + th2 + 3 * pad + 2

        lx1 = x1
        ly1 = y1 - box_h
        if ly1 < 0:
            ly1 = y2
        lx2 = lx1 + box_w
        ly2 = ly1 + box_h

        cv2.rectangle(annotated, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(annotated, label_top, (lx1+pad, ly1+pad+th1),
                    font, font_scale, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(annotated, label_bot, (lx1+pad, ly2-pad),
                    font, font_scale, (255,255,255), 1, cv2.LINE_AA)

        summary[cfg["name"]] += 1

        rec = {
            "cls":    cls_id,
            "name":   cfg["name"],
            "conf":   round(conf, 3),
            "length_px": round(m["length"], 1),
            "width_px":  round(m["width"],  1),
            "area_px":   round(m["area"],   1),
        }
        if scale:
            rec["length_mm"] = round(m["length"] * scale, 1)
            rec["width_mm"]  = round(m["width"]  * scale, 1)
            rec["area_mm2"]  = round(m["area"]   * scale * scale, 1)
        measures.append(rec)

    return annotated, summary, measures


# ── Panel de resumen ───────────────────────────────────────────────────────────

def add_summary_overlay(image, summary, elapsed_ms):
    result = image.copy()
    lines = [
        "DETECCIONES:",
        f"  Rayadura: {summary.get('Rayadura', 0):3d}",
        f"  Agujero:  {summary.get('Agujero',  0):3d}",
        f"  TOTAL:    {sum(summary.values()):3d}",
        f"  {elapsed_ms:.0f} ms",
    ]
    font_scale = 0.55
    font   = cv2.FONT_HERSHEY_SIMPLEX
    pad    = 8
    line_h = 22
    panel_w = 185
    panel_h = len(lines) * line_h + 2 * pad
    overlay = result.copy()
    cv2.rectangle(overlay, (pad, pad), (pad+panel_w, pad+panel_h), (30,30,30), -1)
    result = cv2.addWeighted(overlay, 0.70, result, 0.30, 0)
    for i, line in enumerate(lines):
        color = (255,255,255)
        if "TOTAL" in line: color = (220,220,220)
        cv2.putText(result, line, (pad*2, pad*2 + i*line_h),
                    font, font_scale, color, 1, cv2.LINE_AA)
    return result


# ── Escala px→mm ───────────────────────────────────────────────────────────────

def detect_tile_px(image_bgr):
    """
    Detecta el contorno de la baldosa (rectangulo claro sobre fondo oscuro).
    Devuelve (width_px, height_px) del rectangulo minimo que encierra la baldosa,
    o None si no se puede detectar con confianza.
    """
    H, W = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold: la baldosa es mucho mas clara que el fondo negro
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # El contorno mas grande deberia ser la baldosa
    tile_cnt = max(contours, key=cv2.contourArea)
    tile_area = cv2.contourArea(tile_cnt)

    # Debe ocupar al menos el 30% de la imagen para ser la baldosa
    if tile_area < H * W * 0.30:
        return None

    # Rectangulo minimo que encierra el contorno
    _, (rw, rh), _ = cv2.minAreaRect(tile_cnt)
    return float(max(rw, rh)), float(min(rw, rh))


def compute_scale(image_bgr, tile_cm):
    """
    Calcula mm/px usando el contorno real de la baldosa detectado en la imagen.
    Si no se puede detectar el contorno, usa min(H, W) como fallback.
    tile_cm: lado de la baldosa cuadrada en centimetros.
    """
    tile_mm = tile_cm * 10
    detected = detect_tile_px(image_bgr)
    if detected:
        tile_px = detected[0]  # lado largo = lado de la baldosa
        print(f"[TILE]  Baldosa detectada: {tile_px:.0f}px → {tile_mm:.0f}mm")
        return tile_mm / tile_px
    else:
        H, W = image_bgr.shape[:2]
        px_side = min(H, W)
        print(f"[TILE]  Contorno no detectado, usando min(H,W)={px_side}px como ref")
        return tile_mm / px_side


# ── Inferencia por imagen ──────────────────────────────────────────────────────

def run_inference(model, image_path, conf, iou, imgsz, device,
                  show, save, scale):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  !! No se pudo cargar: {image_path.name}")
        return {}

    t0 = time.perf_counter()
    results = model.predict(source=str(image_path), conf=conf, iou=iou,
                            imgsz=imgsz, device=device, verbose=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    result = results[0]
    annotated, summary, measures = draw_detections(image_bgr, result, scale)
    annotated = add_summary_overlay(annotated, summary, elapsed_ms)

    if show:
        H = image_bgr.shape[0]
        disp = cv2.resize(annotated, None, fx=800/H, fy=800/H) if H > 800 else annotated
        cv2.imshow(f"Rayaduras: {image_path.name}", disp)
        key = cv2.waitKey(0 if len(results) == 1 else 800)
        if key == ord("q"):
            cv2.destroyAllWindows(); sys.exit(0)

    if save:
        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _OUTPUT_DIR / f"pred_{image_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

    # Log por imagen
    total = sum(summary.values())
    print(f"  {image_path.name:50s}  total={total:3d}  {elapsed_ms:.0f}ms")

    # Tabla de medidas por defecto
    unit = "mm" if scale else "px"
    for r in measures:
        if r["cls"] == 1:
            l = r.get("length_mm", r["length_px"])
            w = r.get("width_mm",  r["width_px"])
            print(f"    Rayadura {r['conf']:.0%}  L={l:.1f}{unit}  G={w:.1f}{unit}")
        else:
            lv = r.get("length_mm", r["length_px"])
            wv = r.get("width_mm",  r["width_px"])
            av = r.get("area_mm2",  r["area_px"])
            u2 = "mm²" if scale else "px²"
            print(f"    Agujero  {r['conf']:.0%}  {lv:.1f}x{wv:.1f}{unit}  A={av:.0f}{u2}")

    return {"path": str(image_path), "detections": dict(summary),
            "ms": elapsed_ms, "measures": measures}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    images = collect_images(args.image)
    model_path = find_model(args.model)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics"); sys.exit(1)

    import torch
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")

    print(f"\n[INFER] Cargando {model_path.name} en {device.upper()} ...")
    model = YOLO(str(model_path))

    scale = None
    if args.tile_cm:
        # Se calcula por imagen (depende del tamaño); usamos primera imagen como referencia
        ref = cv2.imread(str(images[0]))
        if ref is not None:
            scale = compute_scale(ref, args.tile_cm)
            print(f"[SCALE] {args.tile_cm}cm → {scale:.4f} mm/px")

    show = not args.no_show
    print(f"[INFER] {len(images)} imagen(es)  conf={args.conf}  iou={args.iou}\n")

    all_results  = []
    total_counts = {}

    for img_path in images:
        r = run_inference(model, img_path, conf=args.conf, iou=args.iou,
                          imgsz=args.imgsz, device=device,
                          show=show, save=args.save, scale=scale)
        if r:
            all_results.append(r)
            for k, v in r.get("detections", {}).items():
                total_counts[k] = total_counts.get(k, 0) + v

    if show:
        cv2.destroyAllWindows()

    # Resumen global
    print("\n" + "=" * 60)
    print("  RESUMEN GLOBAL")
    print("=" * 60)
    print(f"  Imagenes procesadas: {len(all_results)}")
    for k, v in total_counts.items():
        print(f"  {k+':':12s} {v:5d}")
    print(f"  Total detecciones:   {sum(total_counts.values()):5d}")
    if all_results:
        avg_ms = sum(r["ms"] for r in all_results) / len(all_results)
        print(f"  Latencia media:      {avg_ms:.0f} ms/imagen")
    if args.save:
        print(f"  Resultados en:       {_OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
