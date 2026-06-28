@echo off
:: ============================================================
::  SentinelAI – Start Backend Server
::  Activates the project venv and launches Flask
:: ============================================================

title SentinelAI Backend

echo.
echo  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗      █████╗ ██╗
echo  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     ██╔══██╗██║
echo  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     ███████║██║
echo  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     ██╔══██║██║
echo  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗██║  ██║██║
echo  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝
echo.
echo  AI Surveillance Operations Center – Backend v2.0 (Production)
echo  ============================================================
echo.

:: Try sentinelai venv first
if exist "D:\Anomaly detection\sentinelai\venv\Scripts\activate.bat" (
    echo [*] Activating SentinelAI virtual environment...
    call "D:\Anomaly detection\sentinelai\venv\Scripts\activate.bat"
) else if exist "D:\Anomaly detection\venv\Scripts\activate.bat" (
    echo [*] Activating project virtual environment...
    call "D:\Anomaly detection\venv\Scripts\activate.bat"
) else (
    echo [!] No venv found – using system Python
)

:: Navigate to backend folder
cd /d "D:\Anomaly detection\sentinelai\backend"

echo [*] Starting SentinelAI Flask backend...
echo [*] Wildlife Model : D:\Anomaly detection\weights\best.pt
echo [*] Anomaly Model  : D:\Anomaly detection\ana_backend\best_model.keras
echo [*] Server URL     : http://localhost:5000
echo [*] Wildlife API   : http://localhost:5000/api/wildlife
echo [*] Anomaly  API   : http://localhost:5000/api/anomaly
echo [*] Video Feed WL  : http://localhost:5000/video_feed/wildlife
echo [*] Video Feed AN  : http://localhost:5000/video_feed/anomaly
echo.
echo  Press Ctrl+C to stop the server.
echo  ============================================================
echo.

python app.py

pause
