@echo off
setlocal EnableDelayedExpansion
title Ceramic Defect Detector — Installer

echo.
echo  ============================================================
echo    Ceramic Defect Detector  ^|  Auto-installer
echo  ============================================================
echo.

:: ── 1. Check Python ────────────────────────────────────────────
echo  [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found.
    echo          Please install Python 3.11+ from https://python.org
    echo          Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  [OK] Python %PYVER% found.
echo.

:: ── 2. Create virtual environment ──────────────────────────────
echo  [2/4] Creating virtual environment (.venv)...
if exist .venv (
    echo  [SKIP] .venv already exists.
) else (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo  [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  [OK] Virtual environment created.
)
echo.

:: ── 3. Install dependencies ─────────────────────────────────────
echo  [3/4] Installing dependencies (this may take a few minutes)...
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip --quiet

:: Detect NVIDIA GPU for CUDA-enabled PyTorch
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo  [GPU] NVIDIA GPU detected — installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
) else (
    echo  [CPU] No NVIDIA GPU detected — installing CPU-only PyTorch...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet
)

pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo  [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo  [OK] All dependencies installed.
echo.

:: ── 4. Download model weights ───────────────────────────────────
echo  [4/4] Checking model weights...
set WEIGHTS_DIR=scratch_detector\outputs\runs\scratch_yolo\weights
if not exist %WEIGHTS_DIR% mkdir %WEIGHTS_DIR%

if not exist %WEIGHTS_DIR%\best.pt (
    echo  [INFO] Model weights not found.
    echo         Download best.pt from the GitHub Releases page and place it at:
    echo         %WEIGHTS_DIR%\best.pt
    echo.
    echo         Alternatively, train your own model:
    echo           cd scratch_detector
    echo           python src/train.py
) else (
    echo  [OK] Model weights found.
)

echo.
echo  ============================================================
echo    Installation complete!
echo  ============================================================
echo.
echo   To start the web application:
echo.
echo     1. Activate the environment:
echo          .venv\Scripts\activate
echo.
echo     2. Start the server:
echo          cd scratch_detector
echo          uvicorn web.app:app --reload --port 8000
echo.
echo     3. Open your browser at:
echo          http://localhost:8000
echo.
pause
