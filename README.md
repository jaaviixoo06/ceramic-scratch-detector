# Detector de Defectos en Baldosas Cerámicas

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8m--seg-ultralytics-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white)
![Licencia](https://img.shields.io/badge/Licencia-MIT-green)

> **Sistema de inspección visual de calidad cerámica mediante inteligencia artificial** — detecta defectos superficiales (rayaduras y agujeros) en tiempo real con segmentación de instancias YOLOv8m, mide sus dimensiones en milímetros y ofrece un panel web completo listo para integración industrial.

---

## Demo

> **Vídeo explicativo completo** — carga de imagen, resultados anotados, exploración del panel interactivo y mediciones reales de defectos.

[![Ver demo en YouTube](https://i.ytimg.com/vi/BXc1dcMT2Ow/maxresdefault.jpg)](https://youtu.be/BXc1dcMT2Ow)

---

## Características

| | |
|---|---|
| **Segmentación de instancias** | YOLOv8m-seg con pesos entrenados a medida sobre 2 clases: `rayadura` y `agujero` |
| **Medición real** | Convierte máscaras en píxeles a milímetros usando el tamaño físico de la baldosa como referencia de escala (`cv2.minAreaRect`) |
| **Panel web interactivo** | Carga → análisis → resultados con imagen anotada, tabla de defectos y visor lightbox |
| **API asíncrona** | Backend FastAPI con patrón fire-and-poll: el frontend consulta `/results/{id}` sin bloquear el servidor |
| **Pipeline semi-supervisado** | Dos rondas de pseudo-etiquetado → 148 imágenes manuales + 2 132 auto-anotadas |
| **Compatible GPU y CPU** | Funciona con CUDA (NVIDIA) o en modo CPU sin configuración adicional |

---

## Instalación rápida

### Opción A — Instalador automático (recomendado)

**Windows**
```bat
install.bat
```

**Linux / macOS**
```bash
chmod +x install.sh && ./install.sh
```

El instalador realiza automáticamente los siguientes pasos:
1. Comprueba la versión de Python instalada
2. Crea un entorno virtual aislado (`.venv`)
3. Detecta si hay GPU NVIDIA e instala la versión de PyTorch correspondiente (CUDA o CPU)
4. Instala todas las dependencias del fichero `requirements.txt`
5. Indica dónde colocar los pesos del modelo entrenado

---

### Opción B — Instalación manual

```bash
# 1. Clonar el repositorio
git clone https://github.com/jaaviixoo06/ceramic-scratch-detector.git
cd ceramic-scratch-detector

# 2. Crear y activar el entorno virtual
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

# 3. Instalar PyTorch (elegir una opción)
# Con CUDA 12.1 (GPU NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# Solo CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar el resto de dependencias
pip install -r requirements.txt
```

---

### Pesos del modelo entrenado

Los pesos entrenados se distribuyen mediante **GitHub Releases** (no se incluyen en el repositorio por su tamaño).

1. Accede a la [página de Releases](https://github.com/jaaviixoo06/ceramic-scratch-detector/releases/latest)
2. Descarga el fichero `best.pt`
3. Colócalo en la siguiente ruta:

```
scratch_detector/
└── outputs/
    └── runs/
        └── scratch_yolo/
            └── weights/
                └── best.pt   ← aquí
```

> También es posible entrenar el modelo desde cero — consulta la sección [Entrenamiento](#entrenamiento).

---

## Ejecución de la aplicación web

```bash
# Desde la raíz del repositorio (con el entorno virtual activo)
cd scratch_detector
uvicorn web.app:app --reload --port 8000
```

Abre **http://localhost:8000** en el navegador.

### Endpoints de la API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/analyze` | Envía una imagen para su análisis |
| `GET` | `/results/{id}` | Recupera los resultados por identificador |
| `GET` | `/static/dashboard.html?id={id}` | Abre el panel visual de resultados |

**Ejemplo con curl:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@foto_baldosa.jpg" \
  -F "tile_cm=60" \
  -F "conf=0.25"
```

---

## Inferencia desde línea de comandos

```bash
cd scratch_detector

# Imagen individual
python src/infer.py --image ruta/baldosa.jpg

# Con escala real (baldosa de 60 × 60 cm → resultado en mm)
python src/infer.py --image ruta/baldosa.jpg --tile-cm 60

# Procesamiento en lote (carpeta)
python src/infer.py --image ruta/carpeta/ --no-show
```

---

## Entrenamiento

```bash
cd scratch_detector

# Entrenamiento con configuración por defecto (YOLOv8m-seg, 150 épocas)
python src/train.py

# Modelo base y número de épocas personalizados
python src/train.py --model yolov8s.pt --epochs 100
```

La configuración del dataset se encuentra en `scratch_detector/configs/dataset.yaml`.  
Los cuadernos de entrenamiento para Google Colab y Kaggle están disponibles en `scratch_detector/`.

---

## Fórmula de medición

Para cada máscara de defecto detectada se calcula el rectángulo mínimo orientado con `cv2.minAreaRect`:

```
L_px = max(ancho, alto)      # eje más largo  → longitud de la rayadura
G_px = min(ancho, alto)      # eje más corto  → grosor de la rayadura

Escala   s = (lado_real_cm × 10) / L_px_baldosa    [mm/px]

L_mm  = L_px  × s
G_mm  = G_px  × s
A_mm² = área_px² × s²
```

Donde `L_px_baldosa` es el lado de la baldosa en píxeles, derivado de la resolución de la imagen y del parámetro `tile_cm` introducido por el usuario.

---

## Rendimiento del modelo

| Métrica | Valor |
|---------|-------|
| Arquitectura | YOLOv8m-seg (27,3 M parámetros) |
| Plataforma de entrenamiento | Kaggle (2× NVIDIA T4, 3,2 h) |
| Imágenes de entrenamiento | 2 280 (148 manuales + 2 132 pseudo-etiquetadas) |
| **Box mAP@50** | **0,806** |
| **Mask mAP@50** | **0,520** |
| Box mAP@50-95 | 0,591 |
| Mask mAP@50-95 | 0,369 |

---

## Estructura del proyecto

```
ceramic-scratch-detector/
│
├── scratch_detector/              # Paquete principal del proyecto
│   ├── src/
│   │   ├── train.py               # Script de ajuste fino YOLOv8
│   │   ├── infer.py               # Inferencia CLI + medición de defectos
│   │   ├── pre_annotate.py        # Auto-etiquetado con SAM
│   │   └── prepare_dataset.py     # Utilidades de preparación del dataset
│   │
│   ├── web/
│   │   ├── app.py                 # Backend FastAPI
│   │   └── static/
│   │       ├── index.html         # Interfaz de carga de imágenes
│   │       └── dashboard.html     # Panel de resultados
│   │
│   ├── scripts/                   # Utilidades de Label Studio y pseudo-etiquetado
│   ├── configs/
│   │   └── dataset.yaml           # Configuración del dataset YOLO
│   ├── colab_train.ipynb          # Cuaderno de entrenamiento en Google Colab
│   ├── kaggle_train.ipynb         # Cuaderno de entrenamiento en Kaggle
│   └── memoria.pdf                # Memoria técnica completa del proyecto
│
├── requirements.txt
├── install.bat                    # Instalador automático para Windows
└── install.sh                     # Instalador automático para Linux/macOS
```

---

## Tecnologías utilizadas

- **[Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)** — segmentación de instancias
- **[FastAPI](https://fastapi.tiangolo.com/)** — backend REST asíncrono
- **[OpenCV](https://opencv.org/)** — procesamiento de imagen y medición
- **[Label Studio](https://labelstud.io/)** — plataforma de anotación (polígonos + revisión de pseudo-etiquetas)
- **[Albumentations](https://albumentations.ai/)** — pipeline de aumento de datos en entrenamiento
- **[Tailwind CSS](https://tailwindcss.com/)** — estilos del frontend

---

## Memoria técnica

La memoria técnica completa del proyecto se incluye en `scratch_detector/memoria.pdf` y abarca:
- Progresión de filtros: Canny/Sobel/Laplaciano sobre monedas → Sobel vs DoG sobre carretera → Blackhat+Otsu sobre baldosas cerámicas
- Estrategia de anotación en dos fases (bounding boxes → máscaras poligonales para rayaduras)
- Pipeline de pseudo-etiquetado semi-supervisado (2 rondas, 2 132 anotaciones aceptadas)
- Historial de entrenamiento: CPU 14 h → Colab (interrumpido) → Kaggle T4×2 (3,2 h, definitivo)
- Arquitectura de la aplicación web y caso de uso industrial

---

## Autor

**Javier Barba Berrocal**  
3.º curso — *Grado en Ingeniería en la Industria Conectada*  
Asignatura: *Sistemas de Percepción y Visión Artificial*

---

## Licencia

MIT — consulta el fichero [LICENSE](LICENSE) para más detalles.
