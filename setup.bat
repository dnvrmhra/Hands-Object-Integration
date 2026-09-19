@echo off
REM ============================================================
REM  ASTRA-HAR YOLO Pipeline — Setup Script
REM  Creates venv, installs dependencies, prints next steps.
REM ============================================================

echo.
echo ============================================================
echo   ASTRA-HAR YOLO Pipeline Setup
echo ============================================================
echo.

REM --- Check Python is available ---
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.9+ and retry.
    pause
    exit /b 1
)

REM --- Create virtual environment ---
echo [1/3] Creating virtual environment in .\venv\ ...
python -m venv venv
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Done.
echo.

REM --- Activate and install ---
echo [2/3] Installing dependencies from requirements.txt ...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed. Check internet connection and retry.
    pause
    exit /b 1
)
echo       Done.
echo.

REM --- Print instructions ---
echo [3/3] Setup complete!
echo.
echo ============================================================
echo   NEXT STEPS  (run these in order inside the venv)
echo ============================================================
echo.
echo   Activate the venv first:
echo     venv\Scripts\activate.bat
echo.
echo   Step 1 - Generate synthetic dataset (800 train + 200 val):
echo     python synthetic_data\generate_dataset.py
echo.
echo   Step 2 - Train YOLOv8n (50 epochs):
echo     python training\train.py
echo.
echo   Step 3 - Evaluate the trained model:
echo     python training\evaluate.py
echo.
echo   Step 4 - Export to ONNX / TorchScript:
echo     python training\export_onnx.py
echo.
echo   Step 5 - Start inference WebSocket server:
echo     cd inference
echo     uvicorn inference_server:app --host 0.0.0.0 --port 8000 --reload
echo.
echo   (Optional) Mock inference - no model needed:
echo     python inference\mock_inference.py
echo.
echo ============================================================
pause
