# AI Enterprise Procurement System (SmartProcure)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SmartProcure** 是一個專為企業採購戰略部打造的 AI 驅動決策輔助平台。藉由引入機器學習預測模型與可解釋性 AI (SHAP)，本系統將傳統的「事後審核」防呆機制，升級為「事前預警與動態推薦」，協助企業極大化採購節省金額 (Savings) 並有效管控供應鏈風險。

## 🌟 核心特色 (Key Features)

系統採用現代化單頁面應用程式 (SPA) 架構，搭配毛玻璃 (Glassmorphism) 深色主題，並包含以下五大核心模組：

1. 📊 **全局戰略總覽 (Overview Dashboard)**
   - 實時監控預估節省金額、平均風險分數、高風險訂單數量等關鍵 KPI。
2. 🔮 **議價空間預測 (Bargain Prediction)**
   - 在實際下單前，輸入供應商、合約類別、預算與數量，AI 自動預測潛在的**節省率 (Savings Pct)** 與預估金額，協助採購人員掌握談判籌碼。
3. ⚠️ **供應商推薦模型 (Supplier Recommendation Model)**
   - 同樣要買 IT Software，哪個供應商比較值得優先考慮？推薦不是只看誰採購金額最高，而是綜合比較節省率、準時率、供應商風險、供應商狀態、ESG、Preferred Supplier 與單一來源風險。
   - 推薦分數設計：   Savings Score 30%：平均 Savings Pct 越高越好。
     Delivery Score 25%：On Time Delivery 準時率越高越好。
     Risk Score 15%：Supplier Risk 越低越好。
     ESG Score 15%：Supplier ESG Score 越高越好。
     Preferred / Status 15%：Preferred Supplier 與 Supplier Status 越佳越好。
     Single Source Penalty -5%：單一來源比例越高，扣分越多。

4. 🏢 **供應商動態分析 (Supplier Analysis & Recommendation)**
   - 整合供應商分級 (Tier)、地區 (Region)、ESG 永續評分與歷史爭議紀錄 (Controversies)。
   - 支援依據不同採購情境 (降本導向、急件採購、合規導向) 動態調整推薦權重。
5. 📄 **報表與日誌稽核 (Reports & Audit Logs)**
   - 一鍵匯出採購週報/月報，並附帶 LLM 生成的決策摘要。
   - **完善的稽核機制**：系統內所有的「更新 (Update)」與「刪除 (Delete)」操作均會被記錄至日誌，支援操作者、時間戳記與資料變更對比 (Before/After) 查詢。

## 🛠️ 技術架構 (Technology Stack)

* **Frontend:** HTML5, Vanilla JavaScript, CSS3 (Glassmorphism UI)
* **Backend / AI Integration (規劃中):** Python (Scikit-Learn, XGBoost, SHAP) / REST API
* **Icons:** Phosphor Icons

> *註：目前的 Demo 前端頁面 (`dashboard.html`) 已為所有動態資料節點預留 `id` (例如 `id="val-est-savings"`)，方便日後與 Python 後端模型或資料庫進行無縫串接。*

## 🚀 快速開始 (Getting Started)

1. Clone 此專案至本地端：
   ```bash
   git clone https://github.com/your-username/ai-enterprise-procurement-system.git
   ```
2. 進入專案目錄：
   ```bash
   cd ai-enterprise-procurement-system/smart-procurement-ui
   ```
3. 雙擊開啟 `dashboard.html`，即可在瀏覽器中預覽 SPA 決策儀表板。

## 📖 文件參考 (Documentation)

* 詳細的系統架構與各分頁欄位定義，請參閱：[`spec.md`](./spec.md)
* 模型調校規則與戰略手冊：[`model_tuning_playbook.md`](./model_tuning_playbook.md)

## 🤝 貢獻指南 (Contributing)

歡迎提交 Pull Request 或開立 Issue 來協助完善這個專案。進行任何資料結構更新或刪除的邏輯修改時，請確保符合 `spec.md` 規定的日誌稽核 (Audit Log) 標準。

## 📄 授權條款 (License)

This project is licensed under the MIT License - see the LICENSE file for details.
