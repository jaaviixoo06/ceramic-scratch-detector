"""
ls_export_to_dataset.py
=======================
Convierte el export YOLO de Label Studio al formato de entrenamiento seg.

Uso:
  1. Exporta en Label Studio: Export -> YOLO with Images -> descarga ZIP
  2. Extrae el ZIP en cualquier carpeta
  3. Ejecuta:

  python scripts/ls_export_to_dataset.py --export-dir ruta/al/export

El script:
  - Convierte bboxes (5 valores) a poligono de 4 puntos (formato seg)
  - Mantiene poligonos existentes tal cual
  - Aplica augmentacion x8
  - Divide train/val (85/15)
  - Genera scratch_data.zip para Colab
"""

import argparse
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

TARGET_SIZE = 640  # redimensiona todas las imagenes a 640x640 (letterbox)


def letterbox(img: np.ndarray, size: int = TARGET_SIZE):
    """
    Redimensiona manteniendo aspect ratio con relleno gris.
    Devuelve (imagen_lb, scale, left_pad, top_pad).
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top  = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, scale, left, top


def transform_labels(labels: list, orig_h: int, orig_w: int,
                     scale: float, left: int, top: int,
                     size: int = TARGET_SIZE) -> list:
    """Ajusta coordenadas normalizadas de poligonos tras letterbox."""
    new_labels = []
    for cls, coords in labels:
        new_coords = []
        for x, y in coords:
            px = x * orig_w * scale + left
            py = y * orig_h * scale + top
            new_coords.append((
                max(0.0, min(1.0, px / size)),
                max(0.0, min(1.0, py / size)),
            ))
        new_labels.append((cls, new_coords))
    return new_labels

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_DATA_DIR    = _PROJECT_DIR / "data"

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VAL_RATIO = 0.15
SEED      = 42


# ── Conversion de formato ─────────────────────────────────────────────────────

def parse_label_line(line: str):
    """
    Devuelve (cls, coords) donde coords es lista de (x,y) normalizados.
    - Bbox (5 vals):  cls cx cy w h  -> convierte a 4 puntos
    - Seg (>5 vals):  cls x1 y1 x2 y2 ... -> usa directamente
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    vals = list(map(float, parts[1:]))

    if len(vals) == 4:
        # Formato bbox: cx cy w h -> 4 esquinas
        cx, cy, bw, bh = vals
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        # Formato segmentacion: x1 y1 x2 y2 ...
        coords = [(vals[i], vals[i+1]) for i in range(0, len(vals)-1, 2)]

    return cls, coords


def read_labels(lbl_path: Path):
    if not lbl_path.exists():
        return []
    labels = []
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parsed = parse_label_line(line)
        if parsed:
            labels.append(parsed)
    return labels


def save_labels(path: Path, labels: list):
    """Guarda en formato YOLO seg: cls x1 y1 x2 y2 ... xn yn"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for cls, coords in labels:
        pts = " ".join(f"{x:.6f} {y:.6f}" for x, y in coords)
        lines.append(f"{cls} {pts}")
    path.write_text("\n".join(lines), encoding="utf-8")


# ── Augmentacion con soporte de poligonos ────────────────────────────────────

def _build_aug():
    try:
        import albumentations as A
    except ImportError:
        print("ERROR: pip install albumentations")
        raise
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
            A.GaussNoise(var_limit=(5, 30), p=0.30),
            A.CLAHE(clip_limit=2.5, p=0.30),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.20),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


_AUG = None


def augment(img_bgr: np.ndarray, labels: list, n: int = 8):
    """
    Augmenta usando keypoints (un punto por vertice de poligono).
    Devuelve lista de (aug_img, aug_labels).
    """
    global _AUG
    if _AUG is None:
        _AUG = _build_aug()

    if not labels:
        return []

    H, W = img_bgr.shape[:2]

    # Aplanar todos los vertices como keypoints
    keypoints = []
    cls_per_point = []   # clase de cada poligono
    pts_per_label = []   # cuantos puntos tiene cada label

    for cls, coords in labels:
        pts_per_label.append(len(coords))
        for x, y in coords:
            keypoints.append((x * W, y * H))
            cls_per_point.append(cls)

    results = []
    for _ in range(n):
        try:
            out = _AUG(image=img_bgr, keypoints=keypoints)
            aug_img = out["image"]
            aug_kps = out["keypoints"]

            if len(aug_kps) != len(keypoints):
                continue

            aH, aW = aug_img.shape[:2]
            aug_labels = []
            idx = 0
            for cls, n_pts in zip([l[0] for l in labels], pts_per_label):
                pts = aug_kps[idx:idx + n_pts]
                idx += n_pts
                norm = [(max(0, min(1, px / aW)), max(0, min(1, py / aH)))
                        for px, py in pts]
                aug_labels.append((cls, norm))
            results.append((aug_img, aug_labels))
        except Exception:
            pass
    return results


# ── Pipeline principal ────────────────────────────────────────────────────────

def process(export_dir: Path, aug_n: int, val_ratio: float):
    img_dir = export_dir / "images"
    lbl_dir = export_dir / "labels"
    if not img_dir.exists():
        img_dir = export_dir
        lbl_dir = export_dir

    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"ERROR: No se encontraron imagenes en {img_dir}")
        return

    print(f"\n[LS EXPORT] {len(images)} imagenes en {img_dir}")

    tmp = _DATA_DIR / "_tmp_ls"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp_img = tmp / "images"
    tmp_lbl = tmp / "labels"
    tmp_img.mkdir(parents=True)
    tmp_lbl.mkdir(parents=True)

    pairs = []
    n_poly = n_bbox = 0

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        labels   = read_labels(lbl_path)

        for cls, coords in labels:
            if len(coords) > 4:
                n_poly += 1
            else:
                n_bbox += 1

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        lb_img, scale, left_pad, top_pad = letterbox(img)
        lb_labels = transform_labels(labels, orig_h, orig_w, scale, left_pad, top_pad)

        dst_img = tmp_img / (img_path.stem + ".jpg")
        dst_lbl = tmp_lbl / f"{img_path.stem}.txt"
        cv2.imwrite(str(dst_img), lb_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        save_labels(dst_lbl, lb_labels)
        pairs.append((dst_img, dst_lbl))

        if lb_labels and aug_n > 0:
            for j, (aug_img, aug_lbs) in enumerate(augment(lb_img, lb_labels, n=aug_n)):
                a_img = tmp_img / f"{img_path.stem}_aug{j:02d}.jpg"
                a_lbl = tmp_lbl / f"{img_path.stem}_aug{j:02d}.txt"
                cv2.imwrite(str(a_img), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                save_labels(a_lbl, aug_lbs)
                pairs.append((a_img, a_lbl))

    print(f"[LABELS] Poligonos: {n_poly}  Bboxes convertidos: {n_bbox}")
    print(f"[AUG]    Total pares: {len(pairs)}")

    random.seed(SEED)
    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * val_ratio))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    print(f"[SPLIT]  Train={len(train_p)}  Val={len(val_p)}")

    for split_name, split_pairs in [("train", train_p), ("val", val_p)]:
        out_img = _DATA_DIR / "images" / split_name
        out_lbl = _DATA_DIR / "labels" / split_name
        shutil.rmtree(out_img, ignore_errors=True)
        shutil.rmtree(out_lbl, ignore_errors=True)
        out_img.mkdir(parents=True)
        out_lbl.mkdir(parents=True)
        for ip, lp in split_pairs:
            shutil.copy2(ip, out_img / ip.name)
            shutil.copy2(lp, out_lbl / lp.name)

    shutil.rmtree(tmp, ignore_errors=True)

    yaml_path = _DATA_DIR / "dataset.yaml"
    yaml_path.write_text(
        "path: /content/scratch_detector/data\n"
        "train: images/train\n"
        "val:   images/val\n\n"
        "nc: 2\n"
        "names:\n"
        "  0: agujero\n"
        "  1: rayadura\n",
        encoding="utf-8",
    )

    zip_path = _PROJECT_DIR / "scratch_data.zip"
    print(f"[ZIP] Creando {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in (_DATA_DIR / "images").rglob("*"):
            zf.write(f, f.relative_to(_PROJECT_DIR))
        for f in (_DATA_DIR / "labels").rglob("*"):
            zf.write(f, f.relative_to(_PROJECT_DIR))
        zf.write(yaml_path, yaml_path.relative_to(_PROJECT_DIR))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"[ZIP] Listo: {zip_path} ({size_mb:.0f} MB)")

    print("\n" + "=" * 55)
    for split in ["train", "val"]:
        n = len(list((_DATA_DIR / "images" / split).glob("*.*")))
        print(f"  {split:5s}: {n} imagenes")
    print(f"\n  ZIP: {zip_path}")
    print("=" * 55)
    print("\nSiguiente: sube scratch_data.zip a Colab y ejecuta colab_train.ipynb\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=str, required=True)
    ap.add_argument("--aug",        type=int, default=8)
    ap.add_argument("--val-ratio",  type=float, default=VAL_RATIO)
    args = ap.parse_args()
    process(Path(args.export_dir), args.aug, args.val_ratio)


if __name__ == "__main__":
    main()
