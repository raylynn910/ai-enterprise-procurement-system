# AI Enterprise Procurement System (SmartProcure)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SmartProcure** 是一個專為企業採購戰略部打造的 AI 驅動決策輔助平台。藉由引入機器學習預測模型與可解釋性 AI (SHAP)，本系統將傳統的「事後審核」防呆機制，升級為「事前預警與動態推薦」，協助企業極大化採購節省金額 (Savings) 並有效管控供應商風險。

## 🌟 核心特色 (Key Features)

系統採用現代化單頁面應用程式 (SPA) 架構，搭配毛玻璃 (Glassmorphism) 深色主題，並包含以下五大核心模組：

1. 📊 **全局戰略總覽 (Overview Dashboard)**
   - 依「年 → 月」下拉式選單篩選期間，即時監控：
     - 當月節省金額（當月進貨淨額含稅－當月預算金額）
     - 當月節省金額百分比（節省金額 ÷ 當月預算金額）
     - 各供應商平均風險分數
     - 當月異常訂單數量、交貨延遲訂單數量
   - 將以上採購績效指標視覺化呈現。
2. 🔮 **議價空間預測 (Savings Prediction)**
   - 下新 PO 單前，依序選擇：採購類別 (Category)、商品項目 (Item Description)、供應商 (Supplier ID + Supplier Name)、合約類型 (Contract ID + Contract Type)，再輸入下訂數量。
   - AI 自動預測**節省率 (Savings Pct)**（非歷史平均價格），並附上判斷說明文字，解釋此次下訂預期節省/超支的原因與金額。
   - 同步預測該筆新 PO 單是否可能構成違規採購 (Maverick Spend)。
3. 👑 **供應商推薦模型 (Supplier Recommendation)**
   - 選擇採購類別 (Category) 與商品項目 (Item Description) 後，依採購情境（成本優先 / 交期優先 / 合規優先）動態調整推薦權重（綜合考量 Savings、OTD、Risk、ESG），顯示 Top N 供應商推薦排序。
   - 附上 150 字內的判斷說明文字，例如：選擇 A 供應商可降低爭議訂單與對帳糾紛風險；選擇 B 供應商可降低採購成本，但需留意交期延誤風險。
4. ⚠️ **風險訂單清單 (Risk List)**
   - 精準抓出超出預算風險、越權採購 (Maverick Spend)、單一來源 (Single Source)、發票核對異常 (Invoice Match Type 為 No Match / 2-Way Match) 等高風險採購單。
5. 📝 **報表與日誌稽核 (Reports & Audit Logs)**
   - 使用者可選擇「週報」或「月（年）報」並設定日期區間（週報 5–14 天；月/年報 1 個月至 1 年，受限於 Kaggle 資料集時間範圍 2022/1/1–2024/12/31），一鍵產出並匯出，附帶 LLM 生成的決策摘要。
   - **週報核心架構**：以「任務執行」與「緊急處理」為主，涵蓋採購訂單處理進度、物料採購追蹤、交期異常事件與應對措施。
   - **月/年報核心架構**：涵蓋核心績效指標 (KPI)（成本節省率、交期達標率 OTD、品質良率）、採購金額與市場趨勢分析、供應商管理與評鑑、下期工作重點規劃。
   - **完整性稽核**：系統內所有更新 (Update) 與刪除 (Delete) 操作皆記錄於日誌，支援追蹤、時間戳記及資料變更對比查詢。

   > 💡 部分報表子項目（如品質退換貨協調進度、市場行情預警等）依賴真實業界系統資料，本專案採用 Kaggle 公開資料集模擬，暫無對應資料來源，留待實際導入時由採購人員填補。

## 🛠️ 技術架構 (Technology Stack)

* **Frontend:** HTML5, Vanilla JavaScript, CSS3 (Glassmorphism UI)
* **Backend / AI Integration (規劃中):** Python (Scikit-Learn, XGBoost, SHAP) / REST API
* **Icons:** Phosphor Icons

> *註：目前的 Demo 前端頁面 (`dashboard.html`) 已為所有動態資料節點預留 `id` (例如 `id="val-est-savings"`)，方便日後與 Python 後端模型或資料庫進行無縫串接。*

## 🚀 快速開始 (Getting Started)

1. Clone 此專案至本地端：
   ```bash
   git clone https://github.com/raylynn910/ai-enterprise-procurement-system.git
   ```
2. 進入專案目錄：
   ```bash
   cd ai-enterprise-procurement-system
   ```
3. **一鍵啟動 (推薦)**：
   - **Windows**：雙擊執行根目錄下的 `start_demo.bat`。
   - **macOS / Linux**：於終端機執行 `./start_demo.sh`（首次執行需先 `chmod +x start_demo.sh` 給予執行權限）。

   腳本會自動建立虛擬環境、安裝相依套件、在背景啟動 FastAPI 後端伺服器，並開啟瀏覽器載入 `dashboard.html` 儀表板。

## 📖 文件參考 (Documentation)

* 詳細的系統架構與各分頁欄位定義，請參閱：[`spec.md`](./spec.md)
* 供應商風險模型設計、驗證方法與審查修正紀錄：[`docs/supplier_risk_model_report.md`](./docs/supplier_risk_model_report.md)

## 🤝 貢獻指南 (Contributing)

歡迎提交 Pull Request 或開立 Issue 來協助完善這個專案。進行任何資料結構更新或刪除的邏輯修改時，請確保符合 `spec.md` 規定的日誌稽核 (Audit Log) 標準。

## 📄 授權條款 (License)

This project is licensed under the MIT License - see the LICENSE file for details.
