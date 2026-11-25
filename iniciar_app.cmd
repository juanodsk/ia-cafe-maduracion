@echo off
setlocal
cd /d "%~dp0"
echo.
echo Abre http://127.0.0.1:8765 en tu navegador.
echo Conserva esta consola abierta. Ctrl+C detiene el servidor.
echo.
".venv\Scripts\python.exe" -m cafe_ml.web_api
if errorlevel 1 pause
