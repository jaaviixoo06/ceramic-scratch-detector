"""
pseudo_label.py
===============
Genera etiquetas automaticas sobre PRUEBA5 usando el modelo entrenado.
Solo acepta predicciones con confianza >= CONF_THRESH (calidad garantizada).
Combina con las etiquetas manuales y prepara un nuevo scratch_data.zip.

Uso:
  python scripts/pseudo_label.py
  python scripts/pseudo_label.py --conf 0.60 --aug 4
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_ROOT_DIR    = _PROJECT_DIR.parent
_PRUEBA5_DIR = _ROOT_DIR / "PRUEBAS" / "PRUEBA5" / "base_images"
_MODEL_PATH  = _PROJECT_DIR / "outputs" / "runs" / "scratch_yolo" / "weights" / "best4.pt"
_DATA_DIR    = _PROJECT_DIR / "data"
_LS_EXPORT   = _DATA_DIR / "_ls_export"   # etiquetas manuales ya procesadas

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_SZ = 640
VAL_RATIO = 0.15
SEED      = 42


# ── Letterbox ─────────────────────────────────────────────────────────────────

def letterbox(img, size=TARGET_SZ):
    h, w   = img.shape[:2]
    scale  = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    res    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top    = (size - nh) // 2
    left   = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = res
    return canvas, scale, left, top


# ── Guardar etiqueta seg ──────────────────────────────────────────────────────

def save_seg_label(path: Path, labels: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, pts in labels:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"{cls} {coords}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Augmentacion ──────────────────────────────────────────────────────────────

def _build_aug():
    import albumentations as A
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,
                               rotate_limit=20, border_mode=cv2.BORDER_REFLECT, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                                 val_shift_limit=25, p=0.5),
            A.CLAHE(clip_limit=2.5, p=0.30),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )

_AUG = None

def augment(img, labels, n=4):
    global _AUG
    if _AUG is None:
        _AUG = _build_aug()
    if not labels:
        return []
    H, W = img.shape[:2]
    keypoints = []
    pts_per   = []
    clss      = []
    for cls, coords in labels:
        pts_per.append(len(coords))
        clss.append(cls)
        for x, y in coords:
            keypoints.append((x * W, y * H))
    results = []
    for _ in range(n):
        try:
            out = _AUG(image=img, keypoints=keypoints)
            kps = out["keypoints"]
            if len(kps) != len(keypoints):
                continue
            aH, aW = out["image"].shape[:2]
            aug_labels = []
            idx = 0
            for cls, np_ in zip(clss, pts_per):
                pts = kps[idx:idx+np_]
                idx += np_
                norm = [(max(0, min(1, px/aW)), max(0, min(1, py/aH))) for px, py in pts]
                aug_labels.append((cls, norm))
            results.append((out["image"], aug_labels))
        except Exception:
            pass
    return results


# ── Cargar etiquetas manuales del export LS ───────────────────────────────────

def load_manual_pairs(ls_export_dir: Path):
    img_dir = ls_export_dir / "images"
    lbl_dir = ls_export_dir / "labels"
    if not img_dir.exists():
        return []
    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        labels   = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cls  = int(parts[0])
                    vals = list(map(float, parts[1:]))
                    if len(vals) == 4:
                        cx, cy, bw, bh = vals
                        x1, y1 = cx-bw/2, cy-bh/2
                        x2, y2 = cx+bw/2, cy+bh/2
                        coords = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                    else:
                        coords = [(vals[i], vals[i+1]) for i in range(0, len(vals)-1, 2)]
                    labels.append((cls, coords))
        pairs.append((img_path, labels))
    return pairs


# ── Pseudo-labeling con YOLO ──────────────────────────────────────────────────

def pseudo_label_images(image_paths: list, conf_thresh: float) -> list:
    from ultralytics import YOLO
    model = YOLO(str(_MODEL_PATH))
    print(f"[PSEUDO] Modelo cargado: {_MODEL_PATH.name}")
    print(f"[PSEUDO] Procesando {len(image_paths)} imagenes (conf>{conf_thresh}) ...")

    pairs     = []
    accepted  = 0
    skipped   = 0

    for i, img_path in enumerate(image_paths, 1):
        if cv2.imread(str(img_path)) is None:
            skipped += 1
            continue
        try:
            results = model.predict(str(img_path), conf=conf_thresh, iou=0.45,
                                    imgsz=640, verbose=False)
        except Exception as e:
            print(f"  SKIP {img_path.name}: {e}")
            skipped += 1
            continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            skipped += 1
            continue

        img_h, img_w = r.orig_shape
        has_masks    = r.masks is not None and len(r.masks) > 0
        labels       = []

        for j, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if has_masks and cls_id == 1:  # rayadura con mascara
                poly   = r.masks.xy[j]
                coords = [(float(px)/img_w, float(py)/img_h) for px, py in poly]
                if len(coords) >= 3:
                    labels.append((cls_id, coords))
            else:
                # bbox → 4-point polygon
                labels.append((cls_id, [
                    (x1/img_w, y1/img_h), (x2/img_w, y1/img_h),
                    (x2/img_w, y2/img_h), (x1/img_w, y2/img_h),
                ]))

        if labels:
            accepted += 1
            pairs.append((img_path, labels))

        if i % 200 == 0 or i == len(image_paths):
            print(f"  {i}/{len(image_paths)}  aceptadas={accepted}  sin_det={skipped}")

    print(f"[PSEUDO] Aceptadas: {accepted}/{len(image_paths)} imagenes")
    return pairs


# ── Pipeline completo ─────────────────────────────────────────────────────────

def process_pair(img_path, labels, tmp_img, tmp_lbl, aug_n):
    """Letterbox + guardar + augmentar. Devuelve lista de (img_dst, lbl_dst)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    orig_h, orig_w = img.shape[:2]
    lb_img, scale, lpad, tpad = letterbox(img)

    # Transformar coordenadas al espacio letterboxed
    lb_labels = []
    for cls, coords in labels:
        new_coords = [
            (max(0, min(1, (x*orig_w*scale+lpad)/TARGET_SZ)),
             max(0, min(1, (y*orig_h*scale+tpad)/TARGET_SZ)))
            for x, y in coords
        ]
        lb_labels.append((cls, new_coords))

    stem    = img_path.stem
    out_img = tmp_img / f"{stem}.jpg"
    out_lbl = tmp_lbl / f"{stem}.txt"
    cv2.imwrite(str(out_img), lb_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    save_seg_label(out_lbl, lb_labels)
    result = [(out_img, out_lbl)]

    if lb_labels and aug_n > 0:
        for j, (aug_img, aug_lbs) in enumerate(augment(lb_img, lb_labels, n=aug_n)):
            a_img = tmp_img / f"{stem}_aug{j:02d}.jpg"
            a_lbl = tmp_lbl / f"{stem}_aug{j:02d}.txt"
            cv2.imwrite(str(a_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            save_seg_label(a_lbl, aug_lbs)
            result.append((a_img, a_lbl))

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf",      type=float, default=0.65,
                    help="Confianza minima para aceptar pseudo-etiqueta (default: 0.65)")
    ap.add_argument("--aug",       type=int,   default=4,
                    help="Augmentaciones por imagen (default: 4)")
    ap.add_argument("--max-pseudo",type=int,   default=2000,
                    help="Maximo de imagenes pseudo-etiquetadas (default: 2000)")
    args = ap.parse_args()

    print("=" * 55)
    print("  PSEUDO-LABELING AUTOMATICO")
    print("=" * 55)
    print(f"  Conf minima:  {args.conf}")
    print(f"  Augmentacion: x{args.aug}")
    print(f"  Max pseudo:   {args.max_pseudo}")
    print()

    # 1. Etiquetas manuales (desde el export de Label Studio)
    manual_pairs = load_manual_pairs(_LS_EXPORT)
    print(f"[MANUAL] {len(manual_pairs)} imagenes con etiquetas manuales")

    # 2. Pseudo-etiquetas sobre PRUEBA5 (excluir las que ya estan en manual)
    manual_stems = {p.stem for p, _ in manual_pairs}
    all_prueba5  = [p for p in sorted(_PRUEBA5_DIR.iterdir())
                    if p.suffix.lower() in IMG_EXTS
                    and p.stem not in manual_stems]
    random.seed(SEED)
    random.shuffle(all_prueba5)
    all_prueba5 = all_prueba5[:args.max_pseudo * 2]  # candidatas

    pseudo_pairs = pseudo_label_images(all_prueba5, args.conf)
    pseudo_pairs = pseudo_pairs[:args.max_pseudo]
    print(f"[PSEUDO] Usando {len(pseudo_pairs)} imagenes pseudo-etiquetadas")

    # 3. Combinar y procesar
    all_pairs = manual_pairs + pseudo_pairs
    random.shuffle(all_pairs)
    print(f"\n[TOTAL] {len(all_pairs)} imagenes  ({len(manual_pairs)} manual + {len(pseudo_pairs)} auto)")

    tmp     = _DATA_DIR / "_tmp_pseudo"
    tmp_img = tmp / "images"
    tmp_lbl = tmp / "labels"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp_img.mkdir(parents=True)
    tmp_lbl.mkdir(parents=True)

    final_pairs = []
    for img_path, labels in all_pairs:
        final_pairs.extend(process_pair(img_path, labels, tmp_img, tmp_lbl, args.aug))

    print(f"[AUG] Total pares tras augmentacion: {len(final_pairs)}")

    # 4. Split train/val
    random.shuffle(final_pairs)
    n_val   = max(10, int(len(final_pairs) * VAL_RATIO))
    val_p   = final_pairs[:n_val]
    train_p = final_pairs[n_val:]

    for split, pairs in [("train", train_p), ("val", val_p)]:
        out_img = _DATA_DIR / "images" / split
        out_lbl = _DATA_DIR / "labels" / split
        shutil.rmtree(out_img, ignore_errors=True)
        shutil.rmtree(out_lbl, ignore_errors=True)
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)
        for ip, lp in pairs:
            shutil.copy2(ip, out_img / ip.name)
            shutil.copy2(lp, out_lbl / lp.name)

    shutil.rmtree(tmp, ignore_errors=True)

    # 5. dataset.yaml
    yaml_path = _DATA_DIR / "dataset.yaml"
    yaml_path.write_text(
        "path: /content/scratch_detector/data\n"
        "train: images/train\nval: images/val\n\n"
        "nc: 2\nnames:\n  0: agujero\n  1: rayadura\n",
        encoding="utf-8",
    )

    # 6. ZIP
    zip_path = _PROJECT_DIR / "scratch_data.zip"
    print(f"[ZIP] Creando {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (_DATA_DIR / "images").rglob("*"):
            zf.write(f, f.relative_to(_PROJECT_DIR))
        for f in (_DATA_DIR / "labels").rglob("*"):
            zf.write(f, f.relative_to(_PROJECT_DIR))
        zf.write(yaml_path, yaml_path.relative_to(_PROJECT_DIR))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n{'='*55}")
    print(f"  Train: {len(train_p)} imagenes")
    print(f"  Val:   {len(val_p)} imagenes")
    print(f"  ZIP:   {zip_path} ({size_mb:.0f} MB)")
    print(f"{'='*55}")
    print("\nSiguiente: sube scratch_data.zip a Colab y reentrena.\n")


if __name__ == "__main__":
    main()
