# AI Enterprise Procurement System (SmartProcure)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SmartProcure** 是一個專為企業採購戰略部打造的 AI 驅動決策輔助平台。藉由引入機器學習預測模型與可解釋性 AI (SHAP)，本系統將傳統的「事後審核」防呆機制，升級為「事前預警與動態推薦」，協助企業極大化採購節省金額 (Savings) 並有效管控供應鏈風險。

## 🌟 核心特色 (Key Features)

系統採用現代化單頁面應用程式 (SPA) 架構，搭配毛玻璃 (Glassmorphism) 深色主題，並包含以下五大核心模組：

1. 📊 **全局戰略總覽 (Overview Dashboard)**
   - 實時監控預估節省金額、平均風險分數、高風險訂單數量，並將採購績效視覺化。
2. 🔮 **議價空間預測 (Savings Prediction)**
   - 在實際下單前，輸入供應商、合約類別、預算與數量，AI 自動預測潛在的**節省率 (Savings Pct)** 並給予分類判斷 (節省/持平/超支)。
3. 👑 **IT Software 供應商推薦模型 (Supplier Recommendation)**
   - 同樣要買 IT Software，顯示 Top N 供應商推薦排序。
   - 支援依據不同採購情境 (降本導向、急件採購、合規優先) 動態調整推薦權重 (綜合考量 Savings、OTD、Risk、ESG)。
4. ⚠️ **風險訂單清單 (Risk List)**
   - 精準抓出超出預算風險、越權採購 (Maverick Spend)、單一來源 (Single Source) 等異常的高風險採購單。
5. 📝 **報表與日誌稽核 (Reports & Audit Logs)**
   - 一鍵匯出採購週報/月報，並附帶 LLM 生成的決策摘要。
   - **完整性稽核**：系統內的所有更新 (Update) 與刪除 (Delete) 操作皆被記錄於日誌，支援追蹤、時間戳記及資料變更對比查詢。

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
