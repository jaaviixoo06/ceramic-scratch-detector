"""
image_server.py
===============
Servidor HTTP con CORS abierto para servir las imagenes de PRUEBA5
a Label Studio. Ejecutar en una terminal y dejar corriendo.

Uso:
  python scripts/image_server.py
  python scripts/image_server.py --dir /ruta/imagenes --port 8082
"""
import argparse
import http.server
import socketserver
from pathlib import Path

_DEFAULT_DIR = (Path(__file__).resolve().parent.parent.parent
                / "PRUEBAS" / "PRUEBA5" / "base_images")

ap = argparse.ArgumentParser()
ap.add_argument("--dir",  type=Path, default=_DEFAULT_DIR)
ap.add_argument("--port", type=int,  default=8082)
args = ap.parse_args()

PORT      = args.port
IMAGE_DIR = args.dir


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(IMAGE_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):
        pass   # silenciar logs para no llenar la terminal


with socketserver.TCPServer(("", PORT), CORSHandler) as httpd:
    httpd.allow_reuse_address = True
    print(f"Servidor de imagenes con CORS en http://localhost:{PORT}/")
    print(f"Sirviendo desde: {IMAGE_DIR}")
    print("Ctrl+C para parar.\n")
    httpd.serve_forever()
