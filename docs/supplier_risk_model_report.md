# 供應商風險模型 — 訓練/測試最終結果報告

> 資料:`dataset/data_poisntcancelled.csv`(4,649 筆 PO、15 家供應商)
> 標籤:`Supplier Risk`(Low 2,733 / Medium 1,602 / High 314;每家供應商等級固定)
> 產出腳本:`training_scripts/supplier_risk_github.py`、`training_scripts/new_supplier_risk_scoring.py`
> 報告日期:2026-07-17

---

## 摘要(一句話)

逐單模型測試集 AUC 達 1.0,但我們用「留供應商出局」審計證明這是**供應商指紋記憶**(準確率崩至 0.51);因此改以供應商層級、可解釋的政策學習模型回答真正的商業問題——「第 16 家新供應商會被分到哪個風險級、為什麼」。在忠實模擬新供應商資訊狀態的留一供應商驗證(每輪只用 14 家訓練、預測被留出的 1 家)下:**入職當下 66.7%、交易 50 筆後 73.3%**,所有誤判皆為相鄰風險等級、零 Low↔High 對調。

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

在此情境下,Tier / ESG 不是洩漏,而是新供應商入職時本來就會有的
評估屬性——模型的任務是**把公司隱含的風險政策學起來,一致且
可解釋地套用到新供應商**。

### 設計(v2 審計修正版)

- 建模單位:供應商層級(15 家 → 15 列畫像)
- 模型:Logistic Regression + StandardScaler(n=15 時簡單模型泛化
  最好,且係數天生可解釋);設計事前固定,不用驗證結果挑模型
- **兩種評分模式**,對應新供應商生命週期的兩個時點:
  - **Day-0 模式**(入職當下,無任何交易):特徵只有 tier + esg
  - **Review 模式**(交易累積後):加上「觀察至今」的脫軌採購率、
    單一來源率
- 驗證:Leave-One-Supplier-Out(LOSO),共 15 輪:
  每輪**留出 1 家、只用其餘 14 家訓練**,再對被留出的那家做預測;
  每家輪流被留出一次。被留出的供應商在該輪訓練中完全不存在,
  對模型而言與「未來才會出現的第 16 家」處境相同。
  **忠實模擬**:Review 模式驗證時,被留出者只拿它「前 50 筆」
  交易的行為率(新供應商不會有全歷史數據)。
- 注意:LOSO 僅為驗證協定(15 個臨時模型用後即棄);
  最終部署的兩顆模型(day0 / review)是以**全部 15 家**重新訓練的。

### LOSO 驗證結果

| 模式 | 準確率 | 相鄰等級誤判 | Low↔High 對調 |
|---|---|---|---|
| Day-0 入職 | **66.7%(10/15)** | 5 | **0** |
| Review(前 50 筆交易後) | **73.3%(11/15)** | 4 | **0** |

敘事:入職當下憑屬性可達 67%;隨交易累積、行為率穩定後升至 73%。
所有錯誤皆為相鄰等級,從未把 Low 判成 High 或反之。

Day-0 模式逐家明細:

| Supplier | 名稱 | 真實 | LOSO 預測 | 結果 |
|---|---|---|---|---|
| SUP-001 | Apex Industrial Supplies | Low | Low | ✅ |
| SUP-002 | TechPro Components | Low | Low | ✅ |
| SUP-003 | GlobalParts Ltd | Medium | Medium | ✅ |
| SUP-004 | FastTrack Logistics | Low | Low | ✅ |
| SUP-005 | EuroBuild Materials | Low | Low | ✅ |
| SUP-006 | Pacific Rim Supplies | Low | Low | ✅ |
| SUP-007 | Nordic Office Solutions | Low | Low | ✅ |
| SUP-008 | Meridian Tech | Medium | High | ↕️ 相鄰 |
| SUP-009 | Delta Engineering | Medium | Low | ↕️ 相鄰 |
| SUP-010 | SunRise Manufacturing | Low | Low | ✅ |
| SUP-011 | Atlantic Raw Materials | High | Medium | ↕️ 相鄰* |
| SUP-012 | Cornerstone Services | Low | Low | ✅ |
| SUP-013 | Quantum Electronics | Medium | Medium | ✅ |
| SUP-014 | Blue Horizon Packaging | Low | Medium | ↕️ 相鄰 |
| SUP-015 | Iron Gate Steel | Medium | High | ↕️ 相鄰 |

- *SUP-011 是全公司唯一 High:留它出局時訓練集無 High 樣本,
  該 fold 結構上不可能預測正確(資料限制,非模型缺陷)

### 可解釋性輸出範例(虛擬第 16 家:Tier 3、ESG 48)

```
[Day-0 入職]
預測風險等級: High   機率: Low=3.5%  Medium=31.7%  High=64.7%
  1. 供應商層級 (Tier)  值=3.00   貢獻=+1.84  ↑ 推向此級
  2. ESG 評分          值=48.00  貢獻=+0.37  ↑ 推向此級

[Review — 交易 60 筆後, 脫軌率 0.55]
預測風險等級: High   機率: Low=2.4%  Medium=39.9%  High=57.7%
  1. 脫軌採購率 (觀察至今)  值=0.55  貢獻=+1.18  ↑ 推向此級
  2. 單一來源率 (觀察至今)  值=0.10  貢獻=+1.02  ↑ 推向此級
  3. 供應商層級 (Tier)     值=3.00  貢獻=+0.63  ↑ 推向此級
  4. ESG 評分             值=48.00 貢獻=+0.42  ↑ 推向此級
```

已接進後端:`POST /api/predict/new-supplier-risk`(給 tier+esg →
Day-0;再給 mav_rate+single_rate → Review;交易 <25 筆自動提醒)。

### 使用注意

1. High 級全公司只有 1 家:新供應商若真屬 High,較可能被判 Medium
   (系統性低估方向)→ 建議所有 Medium 判定加人工複核
2. n=15 的機率輸出未經校準:98% 不代表真的 98%,只看分級與排序
3. 本引擎學的是「公司既有風險政策」的近似,不是客觀風險;
   政策有偏誤,模型會忠實複製
4. 建議定位:輔助決策(給分級 + 機率 + 原因,採購人員複核),
   而非全自動定級

---

## 審計與修正紀錄(v1 → v2)

v1 版評分引擎經完整審計後,發現並修正以下漏洞
(每一項都以數據驗證,非猜測):

| # | 漏洞 | 影響 | 修正 |
|---|---|---|---|
| 1 | **train/serve 不一致**:LOSO 讓被留出者帶「全歷史」行為率受測,但真正的新供應商入職當下沒有交易歷史 | v1 的 73% 混入未來資訊 | 拆成 Day-0 / Review 兩模式;Review 驗證只給被留出者前 50 筆的行為率 |
| 2 | **指紋特徵**:region/local 在 15 家中近乎身分代理(Americas 僅 2 家、Oceania 1 家),「Americas→High」是單一樣本學來的偽規則 | 解釋輸出把身分當原因 | 移除 region/local |
| 3 | **完全共線**:Tier ⇔ Status ⇔ Preferred 是同一變數三種寫法(Tier1=Preferred=優先、Tier2=Approved、Tier3=Conditional) | 解釋輸出三重計同一訊號 | 只保留 tier |
| 4 | **選擇偏誤**:v1 的 73% 是 16 種「特徵×模型」組合中用同一份 LOSO 挑出的最大值 | 樂觀高估 | v2 設計事前固定,數字直接報告 |
| 5 | **填補洩漏**(第一部分腳本):缺失值用全資料中位數填補,測試統計量洩入訓練 | 輕微(指標變動 <0.2%) | 改為只用訓練集中位數 |

修正後的誠實數字:Day-0 66.7% / Review 73.3%(零 Low↔High 對調)。
v1 報過的 73% 與本表 Review 的 73.3% 數字接近純屬巧合——
組成完全不同(v1 含指紋特徵與全歷史行為率)。

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
