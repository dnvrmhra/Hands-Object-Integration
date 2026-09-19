@echo off
echo ======================================================================
echo   ASTRA-HAR: Live Webcam Hand Tracking & Can/Bottle Movement Server
echo ======================================================================
echo Starting FastAPI Live Video Feed on port 8000...
echo Stream URL: http://localhost:8000/video_feed
echo Telemetry WS: ws://localhost:8000/ws/live
echo.
python -m uvicorn inference.inference_server:app --host 0.0.0.0 --port 8000
pause
