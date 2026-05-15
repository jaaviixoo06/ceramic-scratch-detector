# Ceramic Defect Detector

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8m--seg-ultralytics-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> **AI-powered ceramic tile quality inspection** — detects surface defects (scratches and holes) in real time using YOLOv8m instance segmentation, measures their dimensions in millimetres, and exposes a full web dashboard ready for industrial integration.

---

## Demo

> **Full walkthrough video** — upload an image, get annotated results, explore the interactive dashboard, and see real defect measurements.

https://github.com/jaaviixoo06/ceramic-scratch-detector/releases/download/v1.0/video_testing_web.mp4

---

## Features

| | |
|---|---|
| **Instance segmentation** | YOLOv8m-seg with custom-trained weights on 2 classes: `rayadura` (scratch) and `agujero` (hole) |
| **Real measurements** | Converts pixel masks to mm using tile physical size as scale reference (cv2.minAreaRect) |
| **Web dashboard** | Upload → analyse → interactive results with annotated overlay, defect table, and lightbox |
| **Fire-and-poll API** | Async FastAPI backend; the frontend polls `/results/{id}` so large images never block |
| **Industrial-grade pipeline** | Semi-supervised training with two rounds of pseudo-labeling → 148 manual + 2 132 auto-annotated images |
| **GPU / CPU support** | Runs on CUDA (NVIDIA) or CPU out of the box |

---

## Quick Start

### Option A — One-click installer (recommended)

**Windows**
```bat
install.bat
```

**Linux / macOS**
```bash
chmod +x install.sh && ./install.sh
```

The script will:
1. Check your Python version
2. Create an isolated virtual environment (`.venv`)
3. Auto-detect your GPU and install the right PyTorch build
4. Install all dependencies from `requirements.txt`
5. Guide you through placing the model weights

---

### Option B — Manual install

```bash
# 1. Clone the repository
git clone https://github.com/jaaviixoo06/ceramic-scratch-detector.git
cd ceramic-scratch-detector

# 2. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Install PyTorch (choose one)
# CUDA 12.1 (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
pip install -r requirements.txt
```

---

### Model weights

The trained weights are distributed via **GitHub Releases** (not tracked in git due to file size).

1. Go to the [Releases page](https://github.com/jaaviixoo06/ceramic-scratch-detector/releases/latest)
2. Download `best.pt`
3. Place it at:
```
scratch_detector/
└── outputs/
    └── runs/
        └── scratch_yolo/
            └── weights/
                └── best.pt   ← here
```

> Alternatively, train your own model — see [Training](#training).

---

## Running the web application

```bash
# From the repository root (with .venv active)
cd scratch_detector
uvicorn web.app:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

### API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Submit an image for defect analysis |
| `GET` | `/results/{id}` | Retrieve analysis results by ID |
| `GET` | `/static/dashboard.html?id={id}` | Open the visual dashboard |

**Example with curl:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@tile_photo.jpg" \
  -F "tile_cm=60" \
  -F "conf=0.25"
```

---

## Command-line inference

```bash
cd scratch_detector

# Single image
python src/infer.py --image path/to/tile.jpg

# With real-world scale (60 cm × 60 cm tile)
python src/infer.py --image path/to/tile.jpg --tile-cm 60

# Batch (folder)
python src/infer.py --image path/to/folder/ --no-show
```

---

## Training

```bash
cd scratch_detector

# Train with default settings (YOLOv8m-seg, 150 epochs)
python src/train.py

# Custom base model and epochs
python src/train.py --model yolov8s.pt --epochs 100
```

Dataset configuration is in `scratch_detector/configs/dataset.yaml`.  
Training notebooks for Google Colab and Kaggle are available in `scratch_detector/`.

---

## Measurement formula

For each detected defect mask, the bounding rectangle is computed with `cv2.minAreaRect`:

```
L_px = max(width, height)      # longest axis  → crack length proxy
G_px = min(width, height)      # shortest axis → crack thickness proxy

scale  s = (tile_real_cm × 10) / L_px_tile    [mm/px]

L_mm = L_px × s
G_mm = G_px × s
A_mm² = mask_area_px × s²
```

Where `L_px_tile` is the side of the tile in pixels, derived from the image resolution and the user-supplied `tile_cm` parameter.

---

## Model performance

| Metric | Value |
|--------|-------|
| Architecture | YOLOv8m-seg (27.3 M params) |
| Training platform | Kaggle (2× NVIDIA T4, 3.2 h) |
| Training images | 2 280 (148 manual + 2 132 pseudo-labeled) |
| **Box mAP@50** | **0.806** |
| **Mask mAP@50** | **0.520** |
| Box mAP@50-95 | 0.591 |
| Mask mAP@50-95 | 0.369 |

---

## Project structure

```
ceramic-scratch-detector/
│
├── scratch_detector/          # Main project package
│   ├── src/
│   │   ├── train.py           # YOLOv8 fine-tuning script
│   │   ├── infer.py           # CLI inference + defect measurement
│   │   ├── pre_annotate.py    # Auto-labeling with SAM
│   │   └── prepare_dataset.py # Dataset preparation utilities
│   │
│   ├── web/
│   │   ├── app.py             # FastAPI backend
│   │   └── static/
│   │       ├── index.html     # Upload interface
│   │       └── dashboard.html # Results dashboard
│   │
│   ├── scripts/               # Label Studio & pseudo-labeling helpers
│   ├── configs/
│   │   └── dataset.yaml       # YOLO dataset config
│   ├── colab_train.ipynb      # Google Colab training notebook
│   ├── kaggle_train.ipynb     # Kaggle training notebook
│   └── memoria.pdf            # Full technical report (Spanish)
│
├── testing/
│   └── ejercicios_clase/      # Computer vision exercises (filters, edge detection)
│
├── requirements.txt
├── install.bat                # Windows one-click installer
└── install.sh                 # Linux/macOS one-click installer
```

---

## Tech stack

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — instance segmentation
- **[FastAPI](https://fastapi.tiangolo.com/)** — async REST backend
- **[OpenCV](https://opencv.org/)** — image processing & measurement
- **[Label Studio](https://labelstud.io/)** — annotation platform (polygons + pseudo-label review)
- **[Albumentations](https://albumentations.ai/)** — training-time augmentation pipeline
- **[Tailwind CSS](https://tailwindcss.com/)** — frontend styling

---

## Technical report

A full technical report (in Spanish) is included at `scratch_detector/memoria.pdf`, covering:
- Filter progression: Canny/Sobel/Laplacian on coins → Sobel vs DoG on road → Blackhat+Otsu on ceramic tiles
- Two-phase annotation strategy (bounding boxes → polygonal masks for scratches)
- Semi-supervised pseudo-labeling pipeline (2 rounds, 2 132 accepted annotations)
- Training history: CPU 14 h → Colab (interrupted) → Kaggle T4×2 (3.2 h, final)
- Web application architecture and industrial use case

---

## Author

**Javier Barba Berrocal**  
3rd year — *Ingeniería en la Industria Conectada*  
Subject: *Sistemas de Percepción y Visión Artificial*

---

## License

MIT — see [LICENSE](LICENSE) for details.
