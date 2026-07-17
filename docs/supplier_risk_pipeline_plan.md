# 重寫 supplier_risk_github.py — 供應商風險分類 Pipeline

## Context

`training_scripts/supplier_risk_github.py` 是 Colab notebook 匯出的殘骸：依賴 `google.colab`/`display()`、讀不存在的雲端 CSV、`xgb_weighted` 未定義（執行必崩）、SMOTE 用在獨熱編碼後的縮放特徵、對樹模型做無意義雙重縮放。使用者要求砍掉重練。

**已驗證的資料事實**（`dataset/data_poisntcancelled.csv`，4649 筆 × 57 欄）：
- 15 家供應商，每家 `Supplier Risk` 固定：Low 9 家(2733筆) / Medium 5 家(1602筆) / High 1 家(314筆)
- 標籤是由 Supplier Tier + Preferred Supplier + Maverick Spend (+ESG) 規則合成 → 這些欄位必須排除出 X
- ANOVA（供應商層級，n=15）：**所有純行為欄位聚合後皆無訊號**（OTD、延誤、發票/付款狀態、savings… p>0.4）；唯 maverick rate p=0.014、single-source rate p=0.09
- `PO Date` 格式 `dd/mm/yyyy`，跨 3 年 → 可做 as-of 時序聚合

**使用者決策**：Set A（純行為聚合）與 Set B（行為 + maverick/single-source 聚合）兩組特徵並列對照。

## 交付物

重寫 `training_scripts/supplier_risk_github.py`（同檔名、整檔取代），可直接 `python training_scripts/supplier_risk_github.py` 在本機執行（backend/.venv）。圖表與模型輸出到 `training_scripts/output/`（gitignore 不擋 .png/.pkl → 提醒：**不 commit output**，或加到 .gitignore）。

## 前置：環境

```bash
source backend/.venv/bin/activate
pip install lightgbm catboost matplotlib
```
（xgboost 3.3 / sklearn 1.9 / pandas 3.0 / scipy 已在）

## Pipeline 設計（含對使用者規則的專業修正）

### 1. 資料載入與清理
- 讀 `dataset/data_poisntcancelled.csv`，路徑以 `__file__` 錨定 repo root（不依賴 cwd）
- `PO Date` 用 `pd.to_datetime(..., dayfirst=True)` 解析；Yes/No 欄位映射 0/1
- 全域依 `PO Date` 排序（as-of 特徵的前提）

### 2. 特徵工程 — data aggregation（核心，≥6 個新欄位）
全部採 **as-of（expanding window + shift(1)）per Supplier**，模擬「下單當下只知道歷史」，避免供應商指紋直接常數化，也避免用到未來資訊：

| # | 特徵 | 語意 | 組別 |
|---|---|---|---|
| 1 | `hist_po_count` | 該供應商累計交易次數（交易頻率） | A |
| 2 | `hist_otd_rate` | 歷史準時交貨率 | A |
| 3 | `hist_avg_days_late` | 歷史平均延誤天數 | A |
| 4 | `hist_dispute_rate` | 歷史 PO 爭議率（PO Status=Disputed） | A |
| 5 | `hist_inv_overdue_rate` | 歷史發票逾期率 | A |
| 6 | `hist_pay_overdue_rate` | 歷史付款逾期/凍結率 | A |
| 7 | `hist_avg_savings_pct` | 歷史平均節省率 | A |
| 8 | `hist_avg_lead_time` | 歷史平均交期 | A |
| 9 | `days_since_last_po` | 距上次交易間隔天數 | A |
| 10 | `sup_item_diversity` | 歷史採購品項數(nunique to date) | A |
| 11 | `price_premium` | 本單單價 vs 該品項歷史均價之溢價率 | A |
| 12 | `hist_maverick_rate` | 歷史脫軌採購率 | **B only** |
| 13 | `hist_single_source_rate` | 歷史單一來源率 | **B only** |

當前 PO 情境特徵：`Quantity`、`Lead Time Days`、`Discount Pct`、`Category`/`PO Type`/`Payment Terms`（小基數類別→獨熱；CatBoost 走原生 cat features）。
**排除**（洩漏/指紋）：Supplier ID/Name/Country/Region/Lat/Long、Supplier Tier、Supplier Status、Preferred Supplier、ESG Score、當列 Maverick/Single Source 原始旗標。

### 3. 不平衡處理（修正使用者的順序）
- **先切分、後處理不平衡**（只對訓練集）——不是先處理再切分
- Stratified split：train 70% / val 15% / test 15%（val 供 early stopping）
- 用 `compute_sample_weight('balanced')`（XGB/LGBM）與 `class_weights`（CatBoost），**不用 SMOTE**（在聚合+獨熱特徵空間做內插無意義，且本資料為供應商級標籤，SMOTE 只會合成指紋雜訊）
- y 用 LabelEncoder（High/Low/Medium → 0/1/2），報告時轉回文字

### 4. 模型（三家對打，同一套資料協定）
- **XGBoost**：`max_depth=4, min_child_weight=5, gamma=0.5, subsample=0.7, colsample_bytree=0.7, learning_rate=0.05, n_estimators=2000, early_stopping_rounds=50, eval_metric='mlogloss'`
- **LightGBM**：對應參數（`num_leaves=15, min_child_samples=20, feature_fraction=0.7, bagging_fraction=0.7, early stopping`）
- **CatBoost**：`depth=4, l2_leaf_reg=6, subsample=0.7, early stopping`，類別欄用原生 cat_features
- 全部 `random_state=42`

### 5. 評估（Set A × Set B × 3 模型 = 6 組結果矩陣）
每組輸出：
- **AUC-ROC**（multi-class OvR macro）、**Accuracy**、**Log Loss**、macro-F1
- 訓練集 vs 測試集分數並列（監控過擬合差距）
- classification report + 混淆矩陣（存 PNG）
- 特徵重要性 top15（gain，存 PNG）
- 最後印六組總表，標出最佳組合；最佳模型存 `training_scripts/output/best_model.pkl`

### 6. 誠實註記（寫進腳本尾端輸出）
Set A 分數低是**資料的數學上限**（行為欄位與標籤獨立生成，ANOVA 已證），非模型缺陷；Set B 的提升來自 maverick/single-source 聚合（標籤公式輸入的行為化版本）。此結論供專題報告引用。

## 驗證方式

1. `source backend/.venv/bin/activate && python training_scripts/supplier_risk_github.py` 跑通無錯
2. 檢查輸出：六組指標總表印出、`training_scripts/output/` 有混淆矩陣與特徵重要性 PNG
3. 合理性檢查：Set A 各模型 test AUC 應接近 0.5–0.65（隨機~弱）；Set B 應明顯高（≥0.85 預期）；train/test 差距 < ~0.1 表示正則化有效
4. 不 push；改動留在本機 `jeremy` 分支，由使用者決定何時提交

## 不做的事
- 不動 `backend/models/*.pkl`（線上服務用的模型不受影響）
- 不 commit / push
- 不用 SMOTE、不做特徵縮放（純樹模型不需要）
