# 使用官方 Python 輕量級映像檔
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 將依賴清單複製到容器並安裝
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# 由於 import_data.py 需要讀取上一層的 dataset，
# 我們會在 docker-compose 中將整個專案目錄掛載進來，
# 或者在此將所需檔案複製進去 (開發環境推薦用掛載)。
# 這裡先設定好預設的工作目錄為 backend
WORKDIR /app/backend

# 暴露 FastAPI 預設的 8000 port
EXPOSE 8000

# 啟動腳本會透過 docker-compose 執行 (先匯入資料，再啟動 server)
CMD python import_data.py && uvicorn main:app --host 0.0.0.0 --port 8000
