# AI Enterprise Procurement System (SmartProcure)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**SmartProcure** 是一個專為企業採購戰略部打造的 AI 驅動決策輔助平台。藉由引入機器學習預測模型與可解釋性 AI (SHAP)，本系統將傳統的「事後審核」防呆機制，升級為「事前預警與動態推薦」，協助企業極大化採購節省金額 (Savings) 並有效管控供應商風險。

## 🌟 核心特色 (Key Features)

系統採用現代化單頁面應用程式 (SPA) 架構，搭配毛玻璃 (Glassmorphism) 深色主題，並包含以下五大核心模組：

1. 📊 **全局戰略總覽 (Overview Dashboard)**
   - 下拉式選單年，再來是月，實時監控當月節省金額（當月進貨淨額含稅 - 當月預算金額）、實時監控當月節省金額百分比（當月進貨淨額含稅 - 當月預算金額/當月預算金額）、各供應商平均風險分數、當月異常訂單數量、當月交貨延遲訂單
   數量，並將採購績效視覺化。
2. 🔮 **議價空間預測 (Savings Prediction)**
   - 在實際下新PO單前，能有一個下拉式選單裡面可以選擇Category中的類別，接著選擇我們要的商品項目Item Description、下拉式選單選擇供應商（"Supplier ID" + "Supplier Name"），接著選擇合約類型（"Contract 
   ID"+"Contract Type"），接著選擇我們要的商品項目Item Description、，讓用戶輸入下訂的數量(數字)，讓AI自動預測**節省率 (Savings Pct)** （注意：不是歷史平均價格） 並給予判斷說明文字，例如：因為出自何種原因，所
   以這樣下訂可能節省/超支 多少金額。
   - 預測偵測新下的一張 PO單會不會變成違規採購。
3. 👑 **供應商推薦模型 (Supplier Recommendation)**
   - 同樣能有一個下拉式選單裡面可以選擇Category中的類別，接著選擇我們要的商品項目Item Description，支援依據不同採購情境 (成本優先、交期優先、合規優先) 動態調整推薦權重 (綜合考量 Savings、OTD、Risk、ESG)。，顯示 
   Top N 供應商推薦排序。並給予判斷說明文字(150字內)，例如：跟A供應商買，可以避免之後產生的爭議訂單糾紛產生採購
   與會計作業對帳不合的問題。跟B供應商買，可以降低最多成本，但要小心這些交期延誤等說明文字。
4. ⚠️ **風險訂單清單 (Risk List)**
   - 精準抓出超出預算風險、越權採購 (Maverick Spend)、單一來源 (Single Source)、Invoice Match Type中No Match、2-Way Match等異常的高風險採購單。
5. 📝 **報表與日誌稽核 (Reports & Audit Logs)**
   - 讓用戶先選擇「週報」或是「月（年）報」設定一個日期起迄日，週報最小設定5天，最大設定14天，月（年）報最小設定一個月，最大設定一年（因為我們是用kaggle的資料集所以日期只能是2022.1.1-2024.12.31區間），按下產出報表      後，一鍵匯出採購週報/月報，並附帶 LLM 生成的決策摘要。
   - 一、 採購部門【週報】核心架構
   分別以「任務執行」與「緊急處理」為主，讓主管快速掌握本週工作重點。 
   採購進度：
   1.採購訂單（PO）處理數量與總金額。
   2.物料的採購進度追蹤。
   重點與異常事件處理：
   交期異常： 
   哪些供應商交貨延遲。
   哪些物料面臨延遲？已採取何種催貨或替代方案。（空）
   品質與規格問題： 
   進料檢驗（IQC）異常退換貨協調進度。 （空）
   價格波動： 原物料價格異常上漲的應對。（空）
   下週工作計畫：
   待發出之大型採購案計畫。（空）
   預計進行詢價、比價或議價的項目。（空）
   二、 採購部門【月/年報】核心架構
   月/年報應呈現「數據化成果」與「策略分析」，協助管理層評估部門績效與未來決策：
   核心績效指標 (KPI) 總覽
   成本績效： 本月採購總金額、成本節省率（與預算或歷史價格比較）。
   交期達標率 (OTD)： 供應商準時交貨次數 ÷ 總交貨次數 × 100%。
   品質良率： 合格交次數 ÷ 總交次數 × 100%。 （空）
   採購金額與市場分析：
   主要品項採購金額的月度趨勢分析（可利用圖表呈現同比/環比變化）。
   重大市場行情變動或供應鏈風險預警。 （空）
   供應商管理與評鑑：
   供應商交期。 
   供應商品質或配合度評估。（空）
   供應商尋源（Sourcing）進度與新供應商開發成果。（空）
   下月工作重點 （空）
   下月預算規劃與採購策略調整。（空）
   年度合約續約或招標計畫。（空）
   註：以上（空）為本資料集為kaggle上抓的是不可能會有的，因為沒有真實業界資料所以留空，待採購人員填補。
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
   cd ai-enterprise-procurement-system
   ```
3. **一鍵啟動 (推薦)**：
   雙擊執行根目錄下的 `start_demo.bat`，系統會自動在背景啟動 FastAPI 後端伺服器，並開啟瀏覽器載入 `dashboard.html` 儀表板。

## 📖 文件參考 (Documentation)

* 詳細的系統架構與各分頁欄位定義，請參閱：[`spec.md`](./spec.md)
* 模型調校規則與戰略手冊：[`model_tuning_playbook.md`](./model_tuning_playbook.md)

## 🤝 貢獻指南 (Contributing)

歡迎提交 Pull Request 或開立 Issue 來協助完善這個專案。進行任何資料結構更新或刪除的邏輯修改時，請確保符合 `spec.md` 規定的日誌稽核 (Audit Log) 標準。

## 📄 授權條款 (License)

This project is licensed under the MIT License - see the LICENSE file for details.
