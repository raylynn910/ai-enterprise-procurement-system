@echo off
echo ========================================================
echo      Starting SmartProcure Dashboard (Demo Mode)
echo ========================================================
echo.

echo [1/3] Checking Python Virtual Environment...
cd backend
IF NOT EXIST .venv (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt -q

echo.
echo [2/3] Checking Database Initialization...
IF NOT EXIST procurement.db (
    echo Initializing database from CSV...
    python import_data.py
) ELSE (
    echo Database already exists.
)

echo.
echo [3/3] Starting FastAPI Backend Server (Localhost:8000)...
start "Backend Server" cmd /k "cd backend && .venv\Scripts\activate.bat && uvicorn main:app --reload"

echo.
echo Waiting for backend to initialize (3 seconds)...
timeout /t 3 /nobreak > NUL

echo.
echo [4/4] Opening Dashboard UI in default browser...
start "" "%~dp0smart-procurement-ui\dashboard.html"

echo.
echo ========================================================
echo System started successfully!
echo To stop the system, close the Backend Server terminal.
echo ========================================================
pause
