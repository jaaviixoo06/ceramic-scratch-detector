"""
Actualiza el proyecto de Label Studio (id=11) a anotacion por poligono.
Ejecutar con Label Studio corriendo en localhost:8080.

Uso:
  python scripts/update_ls_to_polygon.py
"""
import os
import requests

BASE    = "http://localhost:8080"
PROJECT = 11

REFRESH = os.environ.get("LS_REFRESH_TOKEN") or input("Pega el refresh token de Label Studio: ").strip()

LABEL_CONFIG = """
<View style="display:flex;flex-direction:column;align-items:center;">
  <Image name="image" value="$image" zoom="true" zoomControl="true"
         brightnessControl="true" contrastControl="true"/>
  <PolygonLabels name="polygon_label" toName="image" strokeWidth="3" opacity="0.35">
    <Label value="rayadura" background="#E20000"
           hint="Haz clic siguiendo la rayadura, doble-clic para cerrar"/>
  </PolygonLabels>
  <RectangleLabels name="rect_label" toName="image">
    <Label value="agujero" background="#00B52A"
           hint="Dibuja un rectangulo sobre el agujero o chip"/>
  </RectangleLabels>
</View>"""

access = requests.post(f"{BASE}/api/token/refresh",
                       json={"refresh": REFRESH}).json()["access"]
H = {"Authorization": f"Bearer {access}", "Content-Type": "application/json"}

r = requests.patch(f"{BASE}/api/projects/{PROJECT}",
                   headers=H,
                   json={"label_config": LABEL_CONFIG})

if r.status_code == 200:
    print("OK — Proyecto actualizado a PolygonLabels")
    print("Las anotaciones antiguas (rectángulos) han sido eliminadas.")
    print("Abre http://localhost:8080/projects/11/ y empieza a anotar con polígonos.")
else:
    print(f"ERROR {r.status_code}: {r.text[:200]}")
