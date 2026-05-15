# start_label_studio.ps1
# Arranca Label Studio con acceso a archivos locales

$VENV_PYTHON = "C:\Users\javie\PycharmProjects\VISION_ARTIFICIAL\.venv\Scripts\python.exe"
$PROJECT_DIR = "C:\Users\javie\PycharmProjects\VISION_ARTIFICIAL"

# Configuracion para servir archivos locales
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT   = $PROJECT_DIR

Write-Host ""
Write-Host "========================================================"
Write-Host "  Label Studio - Detector de Rayaduras en Baldosas"
Write-Host "========================================================"
Write-Host ""
Write-Host "  Abriendo en: http://localhost:8080"
Write-Host "  Archivos locales desde: $PROJECT_DIR"
Write-Host ""
Write-Host "  PRIMER USO: crea una cuenta (se guarda solo localmente)"
Write-Host "  Despues ejecuta: setup_label_studio_project.py"
Write-Host ""
Write-Host "  Para parar el servidor: Ctrl+C"
Write-Host "========================================================"
Write-Host ""

$LABEL_STUDIO = "C:\Users\javie\PycharmProjects\VISION_ARTIFICIAL\.venv\Scripts\label-studio.exe"
& $LABEL_STUDIO start --port 8080
