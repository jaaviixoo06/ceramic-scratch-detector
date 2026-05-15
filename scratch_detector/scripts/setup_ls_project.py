"""
setup_ls_project.py
===================
Crea el proyecto en Label Studio e importa las pre-anotaciones.
Ejecutar DESPUES de que Label Studio este corriendo en localhost:8080.

Uso:
  python scripts/setup_ls_project.py --token <API_TOKEN>

El API token lo encuentras en Label Studio:
  Account & Settings (icono de usuario) -> Access Token
"""

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_DIR = _SCRIPT_DIR.parent
_TASKS_JSON  = _PROJECT_DIR / "data" / "pre_annotations" / "label_studio_tasks.json"

LS_URL = "http://localhost:8080"

LABEL_CONFIG = """
<View style="display:flex; flex-direction:column; align-items:center;">
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <RectangleLabels name="label" toName="image" showInline="true">
    <Label value="rayadura" background="#E20000"
           hint="Scratch, crack or any linear defect"/>
    <Label value="agujero"  background="#00B52A"
           hint="Hole, chip or compact dark spot"/>
  </RectangleLabels>
</View>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", type=str, required=True,
                    help="API token de Label Studio (Account Settings -> Access Token)")
    ap.add_argument("--url", type=str, default=LS_URL,
                    help=f"URL de Label Studio (default: {LS_URL})")
    ap.add_argument("--batch", type=int, default=200,
                    help="Imagenes por batch de importacion (default: 200)")
    args = ap.parse_args()

    try:
        from label_studio_sdk import Client
    except ImportError:
        print("ERROR: pip install label-studio-sdk")
        sys.exit(1)

    print(f"\n[LS] Conectando a {args.url} ...")
    ls = Client(url=args.url, api_key=args.token)
    ls.check_connection()
    print("[LS] Conexion OK")

    # Verificar si ya existe el proyecto
    projects = ls.get_projects()
    existing = [p for p in projects if p.params["title"] == "Rayaduras en Baldosas"]
    if existing:
        print(f"[LS] Proyecto existente encontrado (id={existing[0].id})")
        project = existing[0]
    else:
        print("[LS] Creando proyecto ...")
        project = ls.start_project(
            title        = "Rayaduras en Baldosas",
            label_config = LABEL_CONFIG,
            description  = "Deteccion y clasificacion de defectos en baldosas ceramicas.",
        )
        print(f"[LS] Proyecto creado (id={project.id})")

    # Cargar tasks
    if not _TASKS_JSON.exists():
        print(f"ERROR: No se encontro {_TASKS_JSON}")
        print("       Ejecuta primero: python src/pre_annotate.py --sample")
        sys.exit(1)

    with open(_TASKS_JSON, encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"\n[LS] Importando {len(tasks)} tareas en batches de {args.batch} ...")

    imported = 0
    for i in range(0, len(tasks), args.batch):
        batch = tasks[i : i + args.batch]
        project.import_tasks(batch)
        imported += len(batch)
        print(f"  {imported}/{len(tasks)} importadas")

    print(f"\n[LS] Listo. Abre: {args.url}/projects/{project.id}/")
    print()
    print("FLUJO DE REVISION:")
    print("  1. Haz clic en una imagen")
    print("  2. Las cajas naranjas son pre-anotaciones del modelo")
    print("  3. Ajusta cada caja hasta que rodee el defecto exacto")
    print("  4. Añade cajas nuevas para defectos no detectados")
    print("  5. Borra las cajas incorrectas (falsos positivos)")
    print("  6. Pulsa 'Submit' para guardar y pasar a la siguiente")
    print()
    print("EXPORTAR (cuando hayas revisado un lote):")
    print(f"  {args.url}/projects/{project.id}/export")
    print("  Formato: YOLO with Images")
    print()


if __name__ == "__main__":
    main()
