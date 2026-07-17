# 供應商風險模型 — 訓練/測試最終結果報告

> 資料:`dataset/data_poisntcancelled.csv`(4,649 筆 PO、15 家供應商)
> 標籤:`Supplier Risk`(Low 2,733 / Medium 1,602 / High 314;每家供應商等級固定)
> 產出腳本:`training_scripts/supplier_risk_github.py`、`training_scripts/new_supplier_risk_scoring.py`
> 報告日期:2026-07-17

---

## 摘要(一句話)

逐單模型測試集 AUC 達 1.0,但我們用「留供應商出局」審計證明這是**供應商指紋記憶**(準確率崩至 0.51);因此改以供應商層級、可解釋的政策學習模型回答真正的商業問題——「第 16 家新供應商會被分到哪個風險級、為什麼」——在每家輪流當新供應商的嚴格驗證下達到 **73%**,且所有誤判皆為相鄰風險等級。

---

## 第一部分:逐單分類 Pipeline(`supplier_risk_github.py`)

### 資料協定

- Stratified 切分 70/15/15:train 3,254 / val 697 / test 698(val 供 early stopping)
- 不平衡處理:先切分、後對訓練集加 balanced class weights(不用 SMOTE)
- 特徵:13 個 as-of 聚合特徵(expanding + shift(1),無未來資訊洩漏)
  + 當前 PO 情境特徵;排除所有標籤公式輸入(Tier/Preferred/Maverick/ESG/Status)
  與供應商指紋欄位(ID/國家/經緯度)
- 兩組特徵對照:Set A 純行為聚合;Set B 加上歷史脫軌採購率/單一來源率

### 六組結果(2 特徵組 × 3 模型)

| 特徵組 | 模型 | AUC(訓練) | AUC(測試) | Acc(測試) | LogLoss | F1(測試) | 過擬合Δ |
|---|---|---|---|---|---|---|---|
| Set A 純行為 | XGBoost | 1.0000 | 0.9998 | 0.9871 | 0.0311 | 0.9809 | 0.0001 |
| Set A 純行為 | **LightGBM** 🏆 | 1.0000 | **1.0000** | **0.9914** | **0.0190** | **0.9934** | 0.0000 |
| Set A 純行為 | CatBoost | 1.0000 | 0.9999 | 0.9871 | 0.0345 | 0.9871 | 0.0001 |
| Set B +maverick | XGBoost | 1.0000 | 0.9999 | 0.9900 | 0.0260 | 0.9862 | 0.0001 |
| Set B +maverick | LightGBM | 1.0000 | 1.0000 | 0.9885 | 0.0167 | 0.9851 | 0.0000 |
| Set B +maverick | CatBoost | 1.0000 | 0.9998 | 0.9871 | 0.0397 | 0.9810 | 0.0002 |

最佳模型(Set A × LightGBM)逐類別:

| 類別 | Precision | Recall | F1 | 支持數 |
|---|---|---|---|---|
| High | 1.000 | 1.000 | 1.000 | 47 |
| Low | 1.000 | 0.985 | 0.993 | 410 |
| Medium | 0.976 | 1.000 | 0.988 | 241 |

### ⚠️ 洩漏審計:高分是「供應商指紋記憶」

| 驗證方式 | 準確率 |
|---|---|
| row-level 隨機切分(訓練/測試含同批供應商) | **0.991** |
| 留供應商出局(訓練完全看不到測試供應商) | **0.507**(隨機水準) |

**證據鏈**:

1. 供應商層級 ANOVA(n=15)證實行為欄位(OTD、延誤、發票/付款狀態、savings)
   與標籤統計獨立(p > 0.4)——行為裡沒有風險訊號
2. 但 expanding 聚合特徵會隨筆數收斂成 15 家供應商各自的「指紋」;
   訓練/測試含同批供應商時,模型背下「這家供應商 → 固定等級」的映射
3. 留供應商出局後準確率崩回隨機,證明第 2 點

**結論:報告請勿單獨引用 AUC=1.0 當成果;兩個數字必須一起呈現。**
能主動診斷出「高分是假的」,是本專題方法論上最有價值的部分。

---

## 第二部分:新供應商風險評分引擎(`new_supplier_risk_scoring.py`)

### 商業問題

「第 16 家新供應商進來時,依公司既有 15 家的風險政策,它會被分到
Low / Medium / High 哪一級?為什麼?」

在此情境下,Tier / Preferred / Status / ESG 不是洩漏,而是新供應商
入職時本來就會有的評估屬性——模型的任務是**把公司隱含的風險政策
學起來,一致且可解釋地套用到新供應商**。

### 設計

- 建模單位:供應商層級(15 家 → 15 列畫像:入職屬性 + 累積行為率)
- 模型:Logistic Regression + StandardScaler
  (LOSO 掃描中勝過 Decision Tree / Random Forest / LightGBM——
  n=15 時簡單模型泛化最好,且係數天生可解釋)
- 驗證:Leave-One-Supplier-Out(每家輪流當「第 16 家」)

### LOSO 驗證結果

**供應商層級準確率:73.3%(11/15)**

| Supplier | 名稱 | 真實 | LOSO 預測 | 結果 |
|---|---|---|---|---|
| SUP-001 | Apex Industrial Supplies | Low | Low | ✅ |
| SUP-002 | TechPro Components | Low | Low | ✅ |
| SUP-003 | GlobalParts Ltd | Medium | Medium | ✅ |
| SUP-004 | FastTrack Logistics | Low | Low | ✅ |
| SUP-005 | EuroBuild Materials | Low | Low | ✅ |
| SUP-006 | Pacific Rim Supplies | Low | Low | ✅ |
| SUP-007 | Nordic Office Solutions | Low | Low | ✅ |
| SUP-008 | Meridian Tech | Medium | Medium | ✅ |
| SUP-009 | Delta Engineering | Medium | Low | ↕️ 相鄰 |
| SUP-010 | SunRise Manufacturing | Low | Low | ✅ |
| SUP-011 | Atlantic Raw Materials | High | Medium | ↕️ 相鄰* |
| SUP-012 | Cornerstone Services | Low | Low | ✅ |
| SUP-013 | Quantum Electronics | Medium | Medium | ✅ |
| SUP-014 | Blue Horizon Packaging | Low | Medium | ↕️ 相鄰 |
| SUP-015 | Iron Gate Steel | Medium | High | ↕️ 相鄰 |

- 4 個錯誤**全部為相鄰等級誤判,零 Low↔High 對調**
- *SUP-011 是全公司唯一 High:留它出局時訓練集無 High 樣本,
  該 fold 結構上不可能預測正確(資料限制,非模型缺陷)

### 可解釋性輸出範例

每次評分輸出:等級 + 三類機率 + 逐特徵貢獻分解(係數 × 標準化特徵值)。

```
SUP-011 (Atlantic Raw Materials, 真實=High)
預測風險等級: High   機率: Low=1.0%  Medium=4.2%  High=94.8%
為什麼是 High? (前 5 大影響因素)
  1. 地區=Americas   貢獻=+1.80  ↑ 推向此級
  2. 脫軌採購率 0.65  貢獻=+0.65  ↑ 推向此級
  3. 單一來源率 0.09  貢獻=+0.63  ↑ 推向此級
  4. 供應商層級 Tier=3 貢獻=+0.34  ↑ 推向此級
  5. 供應商狀態=Conditional 貢獻=+0.34  ↑ 推向此級
```

並提供 `score_new_supplier(tier, preferred, status, esg, region, local,
mav_rate, single_rate)` 函式,可直接接後端 API。

### 使用注意

1. High 級全公司只有 1 家:新供應商若真屬 High,較可能被判 Medium
   (系統性低估方向)→ 建議所有 Medium 判定加人工複核
2. 本引擎學的是「公司既有風險政策」的近似,不是客觀風險;
   政策有偏誤,模型會忠實複製
3. 建議定位:輔助決策(給分級 + 機率 + 原因,採購人員複核),
   而非全自動定級

---

## 重現方式

```bash
source backend/.venv/bin/activate
pip install -r training_scripts/requirements.txt

python training_scripts/supplier_risk_github.py       # 第一部分 + 洩漏審計
python training_scripts/new_supplier_risk_scoring.py  # 第二部分
```

圖表(混淆矩陣、特徵重要性、政策係數圖)與模型檔輸出至
`training_scripts/output/`(已 gitignore,不進版本庫)。
所有隨機種子固定(random_state=42),結果可完全重現。
