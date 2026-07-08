# 更新日誌 (Changelog)

所有針對後端專案的顯著變更，都會記錄在這份檔案中。

格式參考 [Keep a Changelog](https://keepachangelog.com/)，且版本號遵循 [Semantic Versioning](https://semver.org/)。

## [Unreleased]

### Added
- 建立 `backend_manual.md` 作為後端開發的原則與手冊。
- 建立 `CHANGELOG.md` 用於追蹤後端的變更與更新。
- 定義了資料庫與版本控制的原則，以及後續階段的分段匯報開發流程。
- **[階段二]** 新增 `import_data.py`，用於將 `dataset/data_final.csv` 的資料自動匯入至本地端 SQLite 資料庫 (`procurement.db`) 中。
- **[階段三]** 建立 `requirements.txt` 來管理後端套件依賴 (FastAPI, Uvicorn, Pandas 等)。
- **[階段三]** 初始化後端框架 `main.py` (FastAPI)，並完成第一支核心 API `GET /api/procurements`，支援分頁與跨網域(CORS)存取。
- **[階段四]** 於 `main.py` 中新增 `POST /api/predict/supplier-risk` Mock API，以供前端開發「供應商風險評估」UI 使用。
- **[階段五]** 新增 `backend/Dockerfile` 與專案根目錄的 `docker-compose.yml`，實現後端環境的容器化，包含自動匯入資料與啟動 API 伺服器。

### Changed
- (尚無變更)

### Fixed
- (尚無修復)
