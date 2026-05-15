"""
prepare_dataset.py  (v2 ? dataset PRUEBA5)
==========================================
Prepara el dataset de rayaduras en baldosas a partir de las im?genes
de PRUEBA5/base_images y su fichero _classes.csv.

Estrategia:
  ? Positivos  -> scratch=1 OR crack=1 (tratados igual para este modelo)
  ? Negativos  -> normal=1 AND scratch=0 AND crack=0 (muestra aleatoria)
  ? Detecci?n  -> OpenCV r?pido (default) o SAM completo (--mode sam)

Clases YOLO:
  0 -> Rayadura_Grande   (bbox_area / img_area >= THRESH_LARGE)
  1 -> Rayadura_Normal   (THRESH_SMALL <= ratio < THRESH_LARGE)
  2 -> Rayadura_Peque?a  (ratio < THRESH_SMALL)

Uso:
  python src/prepare_dataset.py                         # modo CV, dataset PRUEBA5
  python src/prepare_dataset.py --aug 6 --neg 150       # m?s augments y negativos
  python src/prepare_dataset.py --mode sam              # SAM (lento pero m?s preciso)
  python src/prepare_dataset.py --mode manual \         # etiquetas YOLO externas
      --source-images ruta/imgs --source-labels ruta/lbls
"""

import argparse
import csv
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# ?? Rutas base ????????????????????????????????????????????????????????????????
_SCRIPT_DIR  = Path(__file__).resolve().parent          # scratch_detector/src/
_PROJECT_DIR = _SCRIPT_DIR.parent                       # scratch_detector/
_PRUEBAS_DIR = _PROJECT_DIR.parent / "PRUEBAS"

BASE_IMAGES_DIR = _PRUEBAS_DIR / "PRUEBA5" / "base_images"
CSV_PATH        = BASE_IMAGES_DIR / "_classes.csv"
SAM_CHECKPOINT  = _PRUEBAS_DIR / "PRUEBA3" / "weights" / "sam_vit_b_01ec64.pth"

# ?? Umbrales de clasificaci?n por tama?o ?????????????????????????????????????
THRESH_SMALL = 0.003   # < 0.30 % del ?rea imagen -> Peque?a
THRESH_LARGE = 0.020   # >= 2.00 % del ?rea imagen -> Grande

CLASS_NAMES = {0: "Rayadura_Grande", 1: "Rayadura_Normal", 2: "Rayadura_Peque?a"}

# ?? Splits ????????????????????????????????????????????????????????????????????
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10
RANDOM_SEED = 42

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ?? Par?metros SAM ???????????????????????????????????????????????????????????
SAM_PARAMS = dict(
    points_per_side=16, points_per_batch=32,
    pred_iou_thresh=0.85, stability_score_thresh=0.92,
    stability_score_offset=1.0, box_nms_thresh=0.7,
    crop_n_layers=0, crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1,
    min_mask_region_area=100,
)


# =============================================================================
# LECTURA DEL CSV DE CLASES
# =============================================================================

def read_csv_labels(csv_path: Path) -> dict:
    """
    Lee _classes.csv y devuelve dict {filename: {col: int, ?}}.
    Columnas esperadas: filename, crack, nonCeramic, normal, scratch, stain
    """
    labels = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"].strip()
            labels[fname] = {k: int(v) for k, v in row.items() if k != "filename"}
    return labels


def partition_by_csv(csv_labels: dict, images_dir: Path,
                     max_negatives: int) -> tuple:
    """
    Separa las im?genes en:
      positives -> scratch=1 OR crack=1
      negatives -> normal=1 AND scratch=0 AND crack=0  (muestra hasta max_negatives)

    Devuelve (positives: list[Path], negatives: list[Path]).
    """
    positives, negatives = [], []

    for fname, flags in csv_labels.items():
        img_path = images_dir / fname
        if not img_path.exists():
            continue
        if flags.get("scratch", 0) == 1 or flags.get("crack", 0) == 1:
            positives.append(img_path)
        elif flags.get("normal", 0) == 1 and flags.get("scratch", 0) == 0 \
                and flags.get("crack", 0) == 0:
            negatives.append(img_path)

    # Muestra aleatoria reproducible de negativos
    random.seed(RANDOM_SEED)
    if len(negatives) > max_negatives:
        negatives = random.sample(negatives, max_negatives)

    print(f"[CSV]  Positivos (scratch/crack): {len(positives)}")
    print(f"[CSV]  Negativos (normal):        {len(negatives)} de {max_negatives} solicitados")
    return positives, negatives


# =============================================================================
# DETECCI?N R?PIDA CON OPENCV (modo cv, default)
# =============================================================================

def _classify_bbox(w: int, h: int, img_w: int, img_h: int) -> int:
    rel = (w * h) / (img_w * img_h)
    if rel >= THRESH_LARGE:
        return 0   # Grande
    if rel >= THRESH_SMALL:
        return 1   # Normal
    return 2       # Peque?a


def detect_scratches_cv(image_bgr: np.ndarray) -> list:
    """
    Detecta regiones de rayadura usando OpenCV (muy r?pido).

    Pipeline:
      1. CLAHE sobre L (espacio LAB) para mejorar contraste local
      2. Threshold adaptativo -> m?scara binaria
      3. Cierre morfol?gico con kernel alargado (conecta fragmentos de raya)
      4. Contornos -> filtrado por tama?o y aspect-ratio

    Devuelve lista de dicts {"cls", "x", "y", "w", "h"}.
    """
    H, W = image_bgr.shape[:2]

    # 1. Mejora de contraste en espacio LAB
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)

    # 2. Suavizado + threshold adaptativo (maneja variaciones de iluminaci?n)
    blur   = cv2.GaussianBlur(l_enhanced, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )

    # 3. Cierre morfol?gico: conecta fragmentos de raya y cierra huecos
    #    Kernel alargado favorece rayaduras lineales frente a ruido circular
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    closed_h = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_h, iterations=2)
    closed_v = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_v, iterations=2)
    mask = cv2.bitwise_or(closed_h, closed_v)

    # Dilataci?n ligera para fusionar rayaduras muy pr?ximas
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, dilate_k, iterations=1)

    # 4. Contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    min_dim = max(4, min(W, H) // 50)   # tama?o m?nimo proporcional a la imagen

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtros de calidad
        if w < min_dim or h < min_dim:
            continue
        # Eliminar blobs casi cuadrados muy peque?os (ruido de textura)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < 1.5 and (w * h) / (W * H) < THRESH_SMALL:
            continue

        cls = _classify_bbox(w, h, W, H)
        detections.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})

    return detections


# =============================================================================
# DETECCI?N CON SAM (modo sam, lento pero m?s preciso)
# =============================================================================

def load_sam(device: str):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    except ImportError:
        print("ERROR: pip install git+https://github.com/facebookresearch/segment-anything.git")
        sys.exit(1)
    if not SAM_CHECKPOINT.exists():
        print(f"ERROR: SAM no encontrado en {SAM_CHECKPOINT}")
        sys.exit(1)
    print(f"[SAM] Cargando en {device} ?")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    gen = sam_model_registry["vit_b"].__class__   # solo para tipado
    from segment_anything import SamAutomaticMaskGenerator
    return SamAutomaticMaskGenerator(sam, **SAM_PARAMS)


def _create_inverse_mask(shape, masks):
    combined = np.zeros(shape[:2], dtype=np.uint8)
    for m in masks:
        combined[m["segmentation"]] = 255
    return cv2.bitwise_not(combined)


def _remove_hough_lines(mask: np.ndarray) -> np.ndarray:
    cleaned = mask.copy()
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(mask, kernel, iterations=1)
    lines   = cv2.HoughLinesP(dilated, 1, np.pi / 180,
                               threshold=50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cleaned, (x1, y1), (x2, y2), 0, 5)
    return cleaned


def detect_scratches_sam(generator, image_bgr: np.ndarray) -> list:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masks     = generator.generate(image_rgb)
    if len(masks) > 300:
        masks = masks[:300]

    inv     = _create_inverse_mask(image_bgr.shape, masks)
    cleaned = _remove_hough_lines(inv)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = image_bgr.shape[:2]
    detections = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 5:
            continue
        cls = _classify_bbox(w, h, W, H)
        detections.append({"cls": cls, "x": x, "y": y, "w": w, "h": h})
    return detections


# =============================================================================
# CONVERSI?N Y GUARDADO YOLO
# =============================================================================

def detections_to_yolo(detections: list, img_w: int, img_h: int) -> list:
    """[{"cls", "x", "y", "w", "h"}] -> [(cls, cx, cy, bw, bh)] normalizados."""
    labels = []
    for d in detections:
        cx = (d["x"] + d["w"] / 2) / img_w
        cy = (d["y"] + d["h"] / 2) / img_h
        bw = d["w"] / img_w
        bh = d["h"] / img_h
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        bw = max(0.001, min(0.999, bw))
        bh = max(0.001, min(0.999, bh))
        labels.append((d["cls"], cx, cy, bw, bh))
    return labels


def save_yolo_label(label_path: Path, yolo_labels: list):
    lines = [f"{int(cls)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
             for cls, cx, cy, bw, bh in yolo_labels]
    label_path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# AUGMENTACI?N BBOX-SAFE
# =============================================================================

def _build_aug_pipeline():
    try:
        import albumentations as A
    except ImportError:
        print("ERROR: pip install albumentations")
        sys.exit(1)
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.4),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10,
                               rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.30, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                 val_shift_limit=20, p=0.4),
            A.GaussNoise(var_limit=(5, 25), p=0.30),
            A.CLAHE(clip_limit=2.0, p=0.30),
            A.ImageCompression(quality_lower=75, quality_upper=100, p=0.20),
        ],
        bbox_params=dict(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.30,
        ),
    )


_AUG = None


def augment_image(image_bgr: np.ndarray, yolo_labels: list, n: int) -> list:
    global _AUG
    if _AUG is None:
        _AUG = _build_aug_pipeline()
    if not yolo_labels:
        return []

    bboxes  = [(cx, cy, bw, bh) for _, cx, cy, bw, bh in yolo_labels]
    classes = [cls for cls, *_ in yolo_labels]
    results = []

    for _ in range(n):
        try:
            out = _AUG(image=image_bgr, bboxes=bboxes, class_labels=classes)
            if not out["bboxes"]:
                continue
            aug_labels = [(c, *bb) for c, bb in zip(out["class_labels"], out["bboxes"])]
            results.append((out["image"], aug_labels))
        except Exception:
            pass
    return results


def augment_negative(image_bgr: np.ndarray, n: int) -> list:
    """Augmenta una imagen negativa (sin bboxes)."""
    global _AUG
    if _AUG is None:
        _AUG = _build_aug_pipeline()
    results = []
    for _ in range(n):
        try:
            out = _AUG(image=image_bgr, bboxes=[], class_labels=[])
            results.append(out["image"])
        except Exception:
            pass
    return results


# =============================================================================
# PROCESADO PRINCIPAL
# =============================================================================

def process_images(positives: list, negatives: list,
                   tmp_img: Path, tmp_lbl: Path,
                   mode: str, aug_pos: int, aug_neg: int,
                   device: str) -> list:
    """
    Procesa positivos y negativos y guarda pares (img, lbl) en tmp.
    Devuelve lista de (img_path, lbl_path).
    """
    generator = None
    if mode == "sam":
        generator = load_sam(device)

    pairs  = []
    t_init = time.time()

    # ?? Positivos ?????????????????????????????????????????????????????????????
    print(f"\n[POSITIVOS] Procesando {len(positives)} im?genes ?")
    no_detect = 0

    for i, img_path in enumerate(positives, 1):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        H, W = image.shape[:2]

        if mode == "sam":
            t0 = time.time()
            dets = detect_scratches_sam(generator, image)
            print(f"  [{i:04d}/{len(positives)}] {img_path.name}  "
                  f"{len(dets)} det. ({time.time()-t0:.1f}s)")
        else:
            dets = detect_scratches_cv(image)

        yolo_labels = detections_to_yolo(dets, W, H)

        if not dets:
            no_detect += 1

        # Guardar original
        stem      = img_path.stem
        out_img_p = tmp_img / img_path.name
        out_lbl_p = tmp_lbl / f"{stem}.txt"
        shutil.copy2(img_path, out_img_p)
        save_yolo_label(out_lbl_p, yolo_labels)
        pairs.append((out_img_p, out_lbl_p))

        # Augmentaci?n
        if yolo_labels and aug_pos > 0:
            for j, (aug_img, aug_lbs) in enumerate(
                    augment_image(image, yolo_labels, n=aug_pos)):
                a_img_p = tmp_img / f"{stem}_aug{j:02d}.jpg"
                a_lbl_p = tmp_lbl / f"{stem}_aug{j:02d}.txt"
                cv2.imwrite(str(a_img_p), aug_img)
                save_yolo_label(a_lbl_p, aug_lbs)
                pairs.append((a_img_p, a_lbl_p))

        # Progreso cada 100 im?genes (modo cv)
        if mode == "cv" and i % 100 == 0:
            elapsed = time.time() - t_init
            print(f"  [{i:04d}/{len(positives)}] "
                  f"{elapsed:.0f}s transcurridos ? {no_detect} sin detecci?n")

    print(f"  OK Positivos terminados. {no_detect}/{len(positives)} sin detecci?n.")

    # ?? Negativos ?????????????????????????????????????????????????????????????
    print(f"\n[NEGATIVOS] Procesando {len(negatives)} im?genes ?")

    for neg_path in negatives:
        image = cv2.imread(str(neg_path))
        if image is None:
            continue
        stem      = neg_path.stem
        out_img_p = tmp_img / neg_path.name
        out_lbl_p = tmp_lbl / f"{stem}.txt"
        shutil.copy2(neg_path, out_img_p)
        save_yolo_label(out_lbl_p, [])   # etiqueta vac?a = tile sano
        pairs.append((out_img_p, out_lbl_p))

        for j, aug_img in enumerate(augment_negative(image, n=aug_neg)):
            a_img_p = tmp_img / f"{stem}_neg_aug{j:02d}.jpg"
            a_lbl_p = tmp_lbl / f"{stem}_neg_aug{j:02d}.txt"
            cv2.imwrite(str(a_img_p), aug_img)
            save_yolo_label(a_lbl_p, [])
            pairs.append((a_img_p, a_lbl_p))

    print(f"  OK Negativos terminados.")
    return pairs


# =============================================================================
# MODO MANUAL (etiquetas YOLO externas)
# =============================================================================

def process_manual(src_images: Path, src_labels: Path,
                   tmp_img: Path, tmp_lbl: Path, aug: int) -> list:
    images = sorted(p for p in src_images.iterdir()
                    if p.suffix.lower() in IMG_EXTENSIONS)
    print(f"\n[MANUAL] {len(images)} im?genes.")
    pairs   = []
    skipped = 0

    for img_path in images:
        lbl_path = src_labels / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            skipped += 1
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue

        out_img_p = tmp_img / img_path.name
        out_lbl_p = tmp_lbl / lbl_path.name
        shutil.copy2(img_path, out_img_p)
        shutil.copy2(lbl_path, out_lbl_p)
        pairs.append((out_img_p, out_lbl_p))

        if aug > 0:
            raw = []
            for line in lbl_path.read_text().strip().splitlines():
                if line.strip():
                    parts = line.split()
                    raw.append((int(parts[0]),
                                float(parts[1]), float(parts[2]),
                                float(parts[3]), float(parts[4])))
            if raw:
                for j, (aug_img, aug_lbs) in enumerate(
                        augment_image(image, raw, n=aug)):
                    a_img_p = tmp_img / f"{img_path.stem}_aug{j:02d}.jpg"
                    a_lbl_p = tmp_lbl / f"{img_path.stem}_aug{j:02d}.txt"
                    cv2.imwrite(str(a_img_p), aug_img)
                    save_yolo_label(a_lbl_p, aug_lbs)
                    pairs.append((a_img_p, a_lbl_p))

    if skipped:
        print(f"  !!  {skipped} im?genes sin etiqueta omitidas.")
    return pairs


# =============================================================================
# SPLIT, YAML Y RESUMEN
# =============================================================================

def split_and_organize(all_pairs: list, data_dir: Path) -> dict:
    if len(all_pairs) < 3:
        print(f"ERROR: Muy pocas muestras ({len(all_pairs)}). A?ade m?s im?genes.")
        sys.exit(1)

    train_val, test_p = train_test_split(
        all_pairs, test_size=TEST_RATIO, random_state=RANDOM_SEED)
    val_frac = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_p, val_p = train_test_split(
        train_val, test_size=val_frac, random_state=RANDOM_SEED)

    splits = {"train": train_p, "val": val_p, "test": test_p}
    for split_name, pairs in splits.items():
        img_dir = data_dir / "images" / split_name
        lbl_dir = data_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, lbl_path in pairs:
            shutil.move(str(img_path), img_dir / img_path.name)
            shutil.move(str(lbl_path), lbl_dir / lbl_path.name)

    print(f"\n[SPLIT] Train={len(train_p)} | Val={len(val_p)} | Test={len(test_p)}")
    return splits


def create_dataset_yaml(project_dir: Path, data_dir: Path) -> Path:
    cfg_dir   = project_dir / "configs"
    cfg_dir.mkdir(exist_ok=True)
    yaml_path = cfg_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# Rayaduras en baldosas ? generado por prepare_dataset.py\n"
        f"path: {data_dir.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: 3\n"
        f"names:\n"
        f"  0: Rayadura_Grande\n"
        f"  1: Rayadura_Normal\n"
        f"  2: Rayadura_Peque?a\n",
        encoding="utf-8",
    )
    print(f"[YAML] {yaml_path}")
    return yaml_path


def print_summary(data_dir: Path, elapsed: float):
    print("\n" + "=" * 64)
    print("  RESUMEN DEL DATASET")
    print("=" * 64)

    total_imgs, total_anns = 0, 0
    for split in ["train", "val", "test"]:
        lbl_dir = data_dir / "labels" / split
        img_dir = data_dir / "images" / split
        n_imgs  = len(list(img_dir.glob("*.*")))
        counter = Counter()
        for f in lbl_dir.glob("*.txt"):
            for line in f.read_text().strip().splitlines():
                if line.strip():
                    counter[int(float(line.split()[0]))] += 1
        n_anns = sum(counter.values())
        total_imgs += n_imgs
        total_anns += n_anns

        print(f"\n  {split.upper():5s} -> {n_imgs:5d} imgs | {n_anns:6d} anotaciones")
        for cid, name in CLASS_NAMES.items():
            bar = "?" * min(counter[cid] // max(n_imgs // 20, 1), 30)
            print(f"    [{cid}] {name:22s}: {counter[cid]:5d}  {bar}")

        if n_anns > 0:
            dom = max(CLASS_NAMES, key=lambda k: counter[k])
            if counter[dom] / n_anns > 0.80:
                print(f"    !!  Desbalanceo: {CLASS_NAMES[dom]} = "
                      f"{counter[dom]/n_anns*100:.0f}% -> train.py aplicar? class_weights")

    print(f"\n  TOTAL: {total_imgs:5d} im?genes | {total_anns:6d} anotaciones")
    print(f"  Tiempo total: {elapsed/60:.1f} min")
    print("=" * 64)


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Preparaci?n del dataset de rayaduras")
    p.add_argument("--mode", choices=["cv", "sam", "manual"], default="cv",
                   help="cv=OpenCV r?pido (default) | sam=SAM preciso | manual=etiquetas YOLO")
    p.add_argument("--source-images", type=str, default=None)
    p.add_argument("--source-labels", type=str, default=None,
                   help="[--mode manual] Carpeta con .txt YOLO")
    p.add_argument("--aug",     type=int, default=4,
                   help="Augmentaciones por imagen positiva (default: 4)")
    p.add_argument("--neg",     type=int, default=200,
                   help="M?ximo de negativos (im?genes normales) (default: 200)")
    p.add_argument("--aug-neg", type=int, default=1,
                   help="Augmentaciones por imagen negativa (default: 1)")
    p.add_argument("--thresh-small", type=float, default=THRESH_SMALL)
    p.add_argument("--thresh-large", type=float, default=THRESH_LARGE)
    p.add_argument("--device",  type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    global THRESH_SMALL, THRESH_LARGE
    THRESH_SMALL = args.thresh_small
    THRESH_LARGE = args.thresh_large

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("=" * 64)
    print("  PREPARACION DEL DATASET - RAYADURAS EN BALDOSAS")
    print("=" * 64)
    print(f"  Modo:        {args.mode.upper()}")
    print(f"  Dispositivo: {device.upper()}")
    print(f"  Aug positiv: {args.aug}x  |  Aug negativ: {args.aug_neg}x")
    print(f"  Negativos:   hasta {args.neg} im?genes normales")
    print(f"  Peque?a < {THRESH_SMALL:.3f} <= Normal < {THRESH_LARGE:.3f} <= Grande\n")

    data_dir = _PROJECT_DIR / "data"
    tmp_img  = data_dir / "_tmp_images"
    tmp_lbl  = data_dir / "_tmp_labels"
    tmp_img.mkdir(parents=True, exist_ok=True)
    tmp_lbl.mkdir(parents=True, exist_ok=True)

    if args.mode == "manual":
        if not args.source_images or not args.source_labels:
            print("ERROR: --mode manual requiere --source-images y --source-labels")
            sys.exit(1)
        pairs = process_manual(
            Path(args.source_images), Path(args.source_labels),
            tmp_img, tmp_lbl, args.aug,
        )
    else:
        src_images = Path(args.source_images) if args.source_images else BASE_IMAGES_DIR
        if not CSV_PATH.exists() and args.mode != "manual":
            print(f"ERROR: CSV no encontrado en {CSV_PATH}")
            sys.exit(1)

        csv_labels  = read_csv_labels(CSV_PATH)
        positives, negatives = partition_by_csv(csv_labels, src_images, args.neg)

        pairs = process_images(
            positives, negatives,
            tmp_img, tmp_lbl,
            mode=args.mode,
            aug_pos=args.aug,
            aug_neg=args.aug_neg,
            device=device,
        )

    if not pairs:
        print("ERROR: No se gener? ning?n par imagen/etiqueta.")
        sys.exit(1)

    split_and_organize(pairs, data_dir)
    shutil.rmtree(tmp_img, ignore_errors=True)
    shutil.rmtree(tmp_lbl, ignore_errors=True)

    yaml_path = create_dataset_yaml(_PROJECT_DIR, data_dir)
    elapsed   = time.time() - t_start
    print_summary(data_dir, elapsed)

    print("\n" + "=" * 64)
    print("  OK  Dataset generado en scratch_detector/data/")
    print("  Siguiente paso:")
    print("    python src/train.py --config configs/dataset.yaml")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
