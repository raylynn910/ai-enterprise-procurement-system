@echo off
echo ========================================================
echo      Starting SmartProcure Dashboard (Demo Mode)
echo ========================================================
echo.

echo [1/2] Starting FastAPI Backend Server (Localhost:8000)...
start "Backend Server" cmd /k "cd backend && uvicorn main:app --reload"

echo.
echo Waiting for backend to initialize (3 seconds)...
timeout /t 3 /nobreak > NUL

echo.
echo [2/2] Opening Dashboard UI in default browser...
start "" "%~dp0smart-procurement-ui\dashboard.html"

echo.
echo ========================================================
echo System started successfully!
echo To stop the system, close the Backend Server terminal.
echo ========================================================
pause
