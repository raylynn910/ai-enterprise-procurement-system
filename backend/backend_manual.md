# 後端開發手冊與規則 (Backend Manual & Guidelines)

這份文件記錄了本專案後端開發的原則、工作流程以及相關規範。未來的開發、更新都必須遵循此手冊，並在 `CHANGELOG.md` 中記錄變更。

## 1. 資料庫與版本控制規則 (Database & Version Control)
*   **Git 不追蹤資料庫實體檔案**：我們**不會**將真實的資料庫檔案或巨量 CSV 直接 Push 到 GitHub 上，因為這會導致儲存庫過於龐大且容易引發資安與衝突問題。
*   **如何同步資料庫？** 
    *   我們會把「建立資料表的程式碼 (Schema / Migration)」以及「匯入 CSV 的程式碼 (Seed Script)」放在 Git 裡面。
    *   其他開發者只要 clone 專案，執行這些腳本，就可以在自己的電腦上建立一模一樣的本地端資料庫。
    *   未來正式上線時，會在雲端 (如 AWS / GCP / Azure) 建立一個雲端資料庫，讓後端程式連線過去。

## 2. 開發階段與匯報機制 (Workflow & Reporting)
根據協作規則，後端開發必須「**分段進行，每完成一個階段就必須報告，確認後才進行下一步**」。
目前規劃的開發階段如下：

*   **階段一：建立後端規則、手冊與日誌 (當前階段)**
    *   建立此 `backend_manual.md` 及 `CHANGELOG.md`。
    *   解答資料庫同步疑問。
*   **階段二：資料庫建置與資料匯入 (Database & ETL)**
    *   決定本地開發資料庫 (例如 SQLite，因不需額外安裝 server 且易於共享，或是 Postgres via Docker)。
    *   撰寫腳本將 `dataset/data_final.csv` 匯入資料庫。
*   **階段三：後端框架建置與基礎 API**
    *   初始化後端專案 (FastAPI/Node.js 等)。
    *   實作取得採購單、供應商等基礎資料的 API。
*   **階段四：模型 Mock API**
    *   與前端溝通格式，開發預測功能的 Mock API。
*   **階段五：系統整合與容器化**
    *   撰寫 Dockerfile 整合開發環境。

## 3. 程式碼規範 (Coding Standards)
*   **API 格式**：所有的 API 回傳格式必須統一為 JSON，並包含狀態碼 (status code)、訊息 (message) 與資料 (data)。
*   **註解與文件**：重要的商業邏輯、資料匯入腳本必須附上註解。API 開發需搭配 Swagger 或自動化文件。
*   **更新日誌**：任何功能的增加、修改、修復，都必須同步更新至 `CHANGELOG.md`。
