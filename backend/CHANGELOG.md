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
- **[功能新增]** 於 `main.py` 中新增 `GET /api/trends/monthly` API，提供「平均節省率 (Average Savings %)」與「準交率 (On-Time Delivery Rate)」的歷史趨勢數據，以供前端繪製採購績效儀表板圖表。
- **[階段六]** 實作動態補水機制，新增 `GET /api/form-options`、`GET /api/context/category`、`GET /api/context/supplier` 三支 API，提供前端動態參考價與供應商戰力卡片。
- **[階段六]** 新增整合版模型預測 API `POST /api/predict/savings`，由後端自動調用資料庫補齊 10 個模型特徵。
- **[AI 整合]** 成功載入模型組提供的 `reg_model.pkl`、`cls_refiner.pkl` 與 `le_dict.pkl`，於 `/api/predict/savings` 中正式啟用真實 XGBoost 模型推理，取代原本的模擬邏輯。
- **[前端開發]** 刪除舊版 `index.html`，統一以 `dashboard.html` 作為唯一的 SPA 決策儀表板入口。
- **[前端開發]** 於 `dashboard.html` 整合 Chart.js，實作「每月採購績效雙軸混合圖表 (Dual-axis Mixed Chart)」，即時串接後端 `GET /api/trends/monthly` 數據，展示節省率(折線)與準交率(長條圖)對比。
- **[自動化]** 於專案根目錄新增 `start_demo.bat` 一鍵啟動腳本，大幅簡化展示流程 (自動啟動後端並開啟瀏覽器載入 UI)。

### Changed
- (尚無變更)

### Fixed
- (尚無修復)
