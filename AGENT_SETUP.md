# Agent Setup Instructions (For AI Assistants)

> **TO THE AI AGENT READING THIS FILE**:
> The user has just cloned this project to a new computer. Your task is to automatically set up the backend environment and start the server for them. Please read the instructions below and execute the necessary terminal commands.

## Environment Context
* **OS**: Windows (PowerShell)
* **Backend Directory**: `backend`
* **Frontend Directory**: `smart-procurement-ui`

## Execution Steps

Please run the following commands on behalf of the user to initialize the project. **Always execute these inside the `backend` directory.**

1. **Create Virtual Environment & Install Dependencies**:
   We use `uv` for fast dependency management. Please run:
   ```powershell
   cd backend
   uv venv --python 3.12 .venv
   uv pip install -r requirements.txt
   ```

2. **Initialize the SQLite Database**:
   The database file (`.db`) is not tracked by Git. You must initialize it by running the import script:
   ```powershell
   cd backend
   .venv\Scripts\python.exe import_data.py
   ```

3. **Start the API Server**:
   Once the database is ready, start the FastAPI server in the background so it doesn't block your subsequent actions:
   ```powershell
   cd backend
   .venv\Scripts\uvicorn.exe main:app --reload
   ```
   *(Note: Use `WaitMsBeforeAsync` or your background task tool if available, so the server runs continuously).*

4. **Notify the User**:
   After the server is successfully running at `http://127.0.0.1:8000`, inform the user that the environment is fully restored. Tell them they can now double-click to open `smart-procurement-ui\dashboard.html` in their browser to view the application with all APIs functioning correctly.

---
*Created by Antigravity AI - Designed for seamless multi-device collaboration.*
