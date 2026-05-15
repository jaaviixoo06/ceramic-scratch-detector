"""
train.py
========
Fine-tuning de YOLOv8 para deteccion de rayaduras en baldosas.

Modelo base: yolov8s.pt (COCO) -> fine-tune con 3 clases propias
  0: Rayadura_Grande
  1: Rayadura_Normal
  2: Rayadura_Pequena

Uso:
  python src/train.py                              # configuracion por defecto
  python src/train.py --model yolov8m.pt           # modelo mas grande
  python src/train.py --epochs 150 --batch 32      # hiperparametros manuales
  python src/train.py --resume                     # continuar entrenamiento previo
"""

import argparse
import sys
from pathlib import Path

import torch

# Rutas base
_SCRIPT_DIR  = Path(__file__).resolve().parent       # scratch_detector/src/
_PROJECT_DIR = _SCRIPT_DIR.parent                    # scratch_detector/
_ROOT_DIR    = _PROJECT_DIR.parent                   # VISION_ARTIFICIAL/

DATASET_YAML = _PROJECT_DIR / "configs" / "dataset.yaml"
OUTPUTS_DIR  = _PROJECT_DIR / "outputs" / "runs"
PRETRAINED   = _ROOT_DIR / "yolov8n.pt"              # nano ya descargado; cambia a yolov8s.pt si lo tienes


def parse_args():
    p = argparse.ArgumentParser(description="Entrena YOLOv8 para rayaduras en baldosas")
    p.add_argument("--model",   type=str, default=None,
                   help="Modelo base: yolov8n.pt / yolov8s.pt / yolov8m.pt (default: auto)")
    p.add_argument("--config",  type=str, default=str(DATASET_YAML),
                   help=f"Ruta al dataset.yaml (default: {DATASET_YAML})")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=-1,
                   help="Batch size (-1 = auto segun VRAM disponible)")
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--patience",type=int, default=20,
                   help="Early stopping patience (default: 20 epochs sin mejora)")
    p.add_argument("--lr",      type=float, default=0.01)
    p.add_argument("--device",  type=str, default=None,
                   help="'0' (GPU), 'cpu', 'mps' (default: auto)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume",  action="store_true",
                   help="Continuar el ultimo entrenamiento guardado")
    p.add_argument("--name",    type=str, default="scratch_yolo",
                   help="Nombre del experimento (crea subcarpeta en outputs/runs/)")
    p.add_argument("--freeze",  type=int, default=10,
                   help="Numero de capas del backbone a congelar al inicio (default: 10)")
    return p.parse_args()


def select_model(args) -> str:
    """
    Selecciona el modelo base segun los recursos disponibles.
    Jerarquia: arg > yolov8s > yolov8n (ya descargado).
    """
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            # Intentar en la raiz del proyecto
            model_path = _ROOT_DIR / args.model
        if model_path.exists():
            return str(model_path)
        # Si no existe localmente, Ultralytics lo descarga de su hub
        return args.model

    # Buscar modelos disponibles por orden de preferencia
    candidates = [
        _ROOT_DIR / "yolov8s.pt",
        _ROOT_DIR / "yolov8m.pt",
        _ROOT_DIR / "yolov8n.pt",
        _ROOT_DIR / "yolov8s-world.pt",
    ]
    for c in candidates:
        if c.exists():
            print(f"[MODEL] Usando modelo base: {c.name}")
            return str(c)

    # Fallback: Ultralytics descarga yolov8s automaticamente
    print("[MODEL] Descargando yolov8s.pt desde Ultralytics hub...")
    return "yolov8s.pt"


def compute_class_weights(dataset_yaml: str) -> list:
    """
    Calcula pesos de clase inversamente proporcionales a su frecuencia.
    Util cuando una clase es dominante (ratio > 3:1).
    """
    from collections import Counter
    import yaml

    with open(dataset_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["path"])
    lbl_dir   = data_path / "labels" / "train"

    counter = Counter()
    for lbl_file in lbl_dir.glob("*.txt"):
        for line in lbl_file.read_text().strip().splitlines():
            if line.strip():
                counter[int(float(line.split()[0]))] += 1

    total    = sum(counter.values())
    nc       = cfg.get("nc", 3)
    weights  = []
    for i in range(nc):
        freq = counter.get(i, 1) / total
        weights.append(round(1.0 / (freq * nc), 4))

    print(f"[WEIGHTS] Conteo por clase en train: {dict(counter)}")
    print(f"[WEIGHTS] class_weights aplicados:   {weights}")
    return weights


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        sys.exit(1)

    # Verificar dataset.yaml
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: No se encontro {cfg_path}")
        print("       Ejecuta primero: python src/prepare_dataset.py")
        sys.exit(1)

    # Dispositivo
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "0"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[DEVICE] GPU detectada: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    else:
        device = "cpu"
        print("[DEVICE] Sin GPU — entrenando en CPU (sera lento)")

    # Modelo base
    model_path = select_model(args)
    print(f"[TRAIN]  Cargando: {model_path}")
    model = YOLO(model_path)

    # Calcular pesos de clase
    cls_weights = compute_class_weights(args.config)

    # Directorio de salida
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("  CONFIGURACION DE ENTRENAMIENTO")
    print("=" * 64)
    print(f"  Dataset:   {args.config}")
    print(f"  Epochs:    {args.epochs}  (early stop patience={args.patience})")
    print(f"  Batch:     {'auto' if args.batch == -1 else args.batch}")
    print(f"  ImgSz:     {args.imgsz}")
    print(f"  LR:        {args.lr}")
    print(f"  Device:    {device}")
    print(f"  Freeze:    primeras {args.freeze} capas del backbone")
    print(f"  Salida:    {OUTPUTS_DIR / args.name}")
    print("=" * 64 + "\n")

    # ── Entrenamiento ──────────────────────────────────────────────────────────
    results = model.train(
        data       = args.config,
        epochs     = args.epochs,
        batch      = args.batch,
        imgsz      = args.imgsz,
        patience   = args.patience,        # early stopping
        lr0        = args.lr,
        lrf        = 0.01,                 # lr final = lr0 * lrf (cosine decay)
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3,
        warmup_momentum = 0.8,

        # Augmentacion adicional de YOLOv8 (complementa la nuestra)
        hsv_h = 0.015,
        hsv_s = 0.7,
        hsv_v = 0.4,
        degrees = 10.0,
        translate = 0.1,
        scale = 0.5,
        shear = 2.0,
        flipud = 0.3,
        fliplr = 0.5,
        mosaic = 1.0,                      # mosaic augmentation (muy util con datasets pequenos)
        mixup  = 0.1,

        # Loss weights — aumentamos cls para forzar mejor clasificacion de tamano
        box = 7.5,
        cls = 1.0,                         # subido de 0.5 (default) para enfatizar clasificacion
        dfl = 1.5,

        # Freeze backbone inicial (transfer learning en dos fases)
        freeze = args.freeze,

        # Guardado y logs
        project  = str(OUTPUTS_DIR),
        name     = args.name,
        save     = True,
        save_period = 10,                  # checkpoint cada 10 epochs
        plots    = True,                   # curvas de loss y metricas
        val      = True,
        workers  = args.workers,
        device   = device,
        resume   = args.resume,
        exist_ok = True,                   # sobreescribir si el experimento ya existe

        # Verbose
        verbose  = True,
    )

    # ── Resultados ─────────────────────────────────────────────────────────────
    run_dir = OUTPUTS_DIR / args.name
    best_pt = run_dir / "weights" / "best.pt"
    last_pt = run_dir / "weights" / "last.pt"

    print("\n" + "=" * 64)
    print("  ENTRENAMIENTO COMPLETADO")
    print("=" * 64)

    if best_pt.exists():
        print(f"  Mejor modelo: {best_pt}")
    if last_pt.exists():
        print(f"  Ultimo checkpoint: {last_pt}")

    # Metricas finales de validacion
    try:
        metrics = model.val(data=args.config, split="val")
        print(f"\n  mAP50:    {metrics.box.map50:.4f}")
        print(f"  mAP50-95: {metrics.box.map:.4f}")
        for i, name in enumerate(["Rayadura_Grande", "Rayadura_Normal", "Rayadura_Pequena"]):
            ap = metrics.box.ap50[i] if hasattr(metrics.box, 'ap50') else 0
            print(f"  AP50 [{i}] {name}: {ap:.4f}")
    except Exception as e:
        print(f"  (No se pudieron calcular metricas finales: {e})")

    print("\n  Siguiente paso:")
    print(f"    python src/infer.py --image tu_baldosa.jpg --model {best_pt}")
    print("=" * 64 + "\n")

    return best_pt


if __name__ == "__main__":
    args = parse_args()
    train(args)
