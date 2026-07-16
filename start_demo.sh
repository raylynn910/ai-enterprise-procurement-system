#!/bin/bash

echo "========================================================"
echo "     Starting SmartProcure Dashboard (Demo Mode)"
echo "========================================================"
echo ""

echo "[1/3] Checking Python Virtual Environment..."
cd backend || exit

# Check if python3 is available, else try python
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "[2/3] Checking Database Initialization..."
if [ ! -f "procurement.db" ]; then
    echo "Initializing database from CSV..."
    $PYTHON_CMD import_data.py
else
    echo "Database already exists."
fi

echo ""
echo "[3/3] Starting FastAPI Backend Server (Localhost:8000)..."
# Start uvicorn in the background
uvicorn main:app --reload &
BACKEND_PID=$!

echo ""
echo "Waiting for backend to initialize (3 seconds)..."
sleep 3

echo ""
echo "[4/4] Opening Dashboard UI in default browser..."
cd ..
# macOS uses 'open', Linux uses 'xdg-open', Windows/GitBash uses 'start'
if command -v open &>/dev/null; then
    open "smart-procurement-ui/dashboard.html"
elif command -v xdg-open &>/dev/null; then
    xdg-open "smart-procurement-ui/dashboard.html"
elif command -v start &>/dev/null; then
    start "smart-procurement-ui/dashboard.html"
else
    echo "Please open smart-procurement-ui/dashboard.html manually in your browser."
fi

echo ""
echo "========================================================"
echo "System started successfully!"
echo "To stop the system, press CTRL+C."
echo "========================================================"

# Wait for the background process to finish or for the user to press Ctrl+C
wait $BACKEND_PID
