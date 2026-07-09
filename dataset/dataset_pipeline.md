# EDA_data_1.ipynb Pipeline 邏輯說明文件

> **檔案來源**：`EDA_data_1.ipynb`（46 cells：22 code / 24 markdown）
> **資料集**：`data.csv`（5,200 rows × 57 columns，結構式模擬資料 / synthetic dataset）
> **兩條主要工作線**：
> 1. **Track A**（Cell 1–33）：以 `Savings Pct` 為目標變數的 EDA，含洩漏欄位處理、目標分佈檢視、類別/數值特徵關係、PCA、MCA。
> 2. **Track B**（Cell 34–45）：供應商推薦模型（Model 2）的資料準備，重新載入原始資料並按 `Category × Supplier Name` 聚合績效。

---

## 一、Pipeline 全景圖

```
┌──────────────────────────────────────────────────────────────────┐
│                     Track A：Savings Pct EDA                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Cell 1  載入套件 + 讀 data.csv → df                                │
│  Cell 2  df.info()                                                 │
│                                                                    │
│  ─────────────── 資料清理（處理 Cancelled 訂單）───────────────      │
│  Cell 4  Cancelled → Actual Delivery = NaN, Savings Pct = NaN      │
│  Cell 5  Cancelled → On Time Delivery = NaN                        │
│  Cell 6–8  檢視欄位                                                 │
│                                                                    │
│  ─────────────── 洩漏欄位處理 ───────────────                        │
│  Cell 10  df_copy_new = df.drop([8 個欄位])                         │
│                                                                    │
│  ─────────────── 目標變數 EDA ───────────────                        │
│  Cell 12  describe                                                 │
│  Cell 13  histplot + KDE                                           │
│  Cell 17–18  類別特徵 vs Savings Pct（Spearman）                    │
│  Cell 20  stripplot                                                │
│  Cell 21  IQR 離群值分析                                            │
│  Cell 23  histplot（不含 KDE）                                      │
│                                                                    │
│  ─────────────── 多變量分析 ───────────────                          │
│  Cell 25–26  數值相關係數矩陣                                       │
│  Cell 28  PCA + Scree Plot                                         │
│  Cell 30  PCA Loadings 熱圖                                         │
│  Cell 33  MCA（類別特徵二維投影）                                    │
│                                                                    │
├──────────────────────────────────────────────────────────────────┤
│                Track B：供應商推薦模型（Model 2）                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Cell 37  重新載入 data.csv → df_original → df_sub                  │
│  Cell 38  df_sub.drop(Cancelled) → 4,649 筆                        │
│                                                                    │
│  ─────────────── 供應商績效聚合 ───────────────                       │
│  Cell 42  按 [Category, Supplier Name] 分組（mean 版本）             │
│  Cell 43  氣泡圖 + Maverick vs Savings 回歸圖                        │
│  Cell 45  按 [Category, Supplier Name] 分組（median 版本）           │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 二、逐 Cell 邏輯說明

### 【區塊 A-1】資料載入與初步檢視

#### Cell 1 — 套件載入與資料讀取
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
df = pd.read_csv('/content/drive/MyDrive/I DI/專題/data.csv')
pd.set_option('display.max_columns', None)
df.head()
```
- **目的**：載入分析工具，讀取 5,200 筆結構式模擬採購資料到 `df`。
- **設定顯示所有欄位**（因為有 57 個欄位，預設會被截斷）。

#### Cell 2 — 資料結構總覽
```python
df.info()
```
- 檢查每個欄位的 dtype 與非空值數，作為後續清理的判斷依據。

---

### 【區塊 A-2】Cancelled 訂單清理

#### Cell 3（markdown）—— 意圖說明
> 「對所有 PO Status 為 Cancelled 的 Actual Delivery 填上空值，Savings Pct 也要為 0」

**⚠️ 邏輯提醒**：Markdown 上寫「Savings Pct 也要為 0」，但 Cell 4 實際做的是設為 `NaN`（不是 0）。文字與程式碼不一致，建議把 markdown 改為「Savings Pct 也要設為空值」以避免日後誤解。從方法論看，**設為 NaN 是正確的**：訂單被取消代表交易未發生，節省率不應該存在（也不是零），NaN 語意才對。

#### Cell 4 — Actual Delivery + Savings Pct 設為 NaN
```python
df.loc[df['PO Status'] == 'Cancelled', 'Actual Delivery'] = pd.NA
df.loc[df['PO Status'] == 'Cancelled', 'Savings Pct'] = pd.NA
```
- **邏輯依據**：Cancelled = 取消，實際交貨與節省率都不存在。
- **影響範圍**：551 筆（已驗證）。

#### Cell 5 — On Time Delivery 設為 NaN
```python
df.loc[df['PO Status'] == 'Cancelled', 'On Time Delivery'] = pd.NA
```
- 同上邏輯：Cancelled 訂單沒有「準時」的概念。
- **注意**：`Days Late` 沒有一起處理（它也是 post-delivery outcome）。若後續要用 `Days Late` 當特徵，這裡應該一併設 NaN。

#### Cell 6–8 — 欄位檢視
- `df.columns.tolist()`、`df.info()`、`df.head()`。
- 沒有修改資料，僅為視覺確認。

---

### 【區塊 A-3】洩漏欄位處理

#### Cell 9（markdown）—— 「刪除洩漏欄位」

#### Cell 10 — 建立乾淨副本 df_copy_new
```python
df_copy_new = df.copy()
columns_to_drop = [
    'Line Total Gross', 'Line Total Inc Tax', 'Budget Unit Price',
    'Budget Total', 'Quantity', 'Savings Amount', 'Line Net', 'Unit Price'
]
df_copy_new.drop(columns=existing_columns_to_drop, inplace=True)
```

- **刪除清單分類**：

| 欄位 | 刪除理由 |
|---|---|
| `Budget Unit Price` | 目標公式的分母（Savings Pct 由它算出，r ≈ 0.993）— **必刪** |
| `Budget Total` | Budget Unit Price × Quantity，同一 formula chain — **必刪** |
| `Savings Amount` | Savings Pct 的姊妹欄（同一分子） — **必刪** |
| `Unit Price` | Savings Pct 的分子基礎 — **必刪** |
| `Line Net` | Unit Price × Quantity × (1 – Discount Pct) — **必刪** |
| `Line Total Gross`, `Line Total Inc Tax` | Line Net 的衍生（含稅版本） — **必刪** |
| `Quantity` | 存在爭議（見下方提醒） |

**⚠️ 提醒（Quantity）**：
Quantity 本身並不是 Savings Pct 的直接構成項——Savings Pct = (Budget Unit Price − Unit Price) / Budget Unit Price × 100，公式裡沒有 Quantity。刪掉 Quantity 會失去一個潛在有意義的特徵：**採購數量本身可能與議價空間有關**（volume discount 通常在 base Unit Price 就反映出來，但採購方在下單時知道量體，這是模型可用的 pre-decision feature）。建議在 Subtopic A 建模階段保留 Quantity。

---

### 【區塊 A-4】目標變數 Savings Pct 的單變量 EDA

#### Cell 11（markdown）—— 「目標變數 Savings Pct 的 EDA」

#### Cell 12 — 敘述統計
```python
print(df['Savings Pct'].describe())
```
- 檢查 mean、std、min、max、四分位數。
- 已驗證：mean ≈ 8.5%、std ≈ 12.9%、min = −33.7%、max = 31.9%、median = 11.7%。

#### Cell 13 — 分佈直方圖（含 KDE）
- 觀察整體形狀、雙峰/單峰、偏度。

#### Cell 14 — 註解說明（不執行實質程式）
- 說明「EDA 階段不預先做對數轉換」的方法論理由。這一段其實是想法紀錄，不是程式邏輯。

#### Cell 15 — `df`（單獨列印 DataFrame）
- **⚠️ 冗餘 cell**：這個 cell 只是 `df`，實務上沒作用，可以刪掉。

---

### 【區塊 A-5】類別型特徵與 Savings Pct 的關係

#### Cell 17 — 列出類別欄位
```python
categorical_columns = df.select_dtypes(include='object').columns.tolist()
```

#### Cell 18 — 類別特徵的雙圖分析（核心 EDA cell）
邏輯步驟：
1. 過濾出 nunique ≤ 60 的類別欄位（可讀性考量）。
2. 對每個欄位畫：
   - **左圖**：`countplot`（頻率分佈）
   - **右圖**：`boxplot`（該類別下 Savings Pct 的分佈）
3. 用 `LabelEncoder` 把類別編碼後計算 **Spearman 相關係數**，最後排序印出。

**⚠️ 方法論提醒**：
- LabelEncoder 會把類別給予任意數值編碼（例如 A→0, B→1, C→2），這個編碼順序是**任意的**，用它算 Spearman 相關係數在方法論上不嚴謹。對名目類別（nominal）而言，應該用 **η²（eta squared）** 或 **ANOVA F 統計量** 才對。目前的分數只能當作粗略排序參考，不能當結論。
- 對序位類別（如 `Supplier Tier` 1/2/3、`Supplier Risk` Low/Medium/High）Spearman 才是合理的。

---

### 【區塊 A-6】Savings Pct 的分佈細節

#### Cell 19（markdown）—— stripplot 說明

#### Cell 20 — stripplot（點狀分佈圖）
- 使用 jitter 顯示每個資料點，看點的密度分佈。

#### Cell 21 — IQR 離群值分析
```python
Q1 = savings_pct_data.quantile(0.25); Q3 = ...
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ...
```
- 標準 Tukey 1.5×IQR 法識別離群值。
- **注意**：這裡只是「識別」，還沒刪除。是否刪除離群值應該在建模前決策，且不同 subtopic 決策可能不同（Subtopic A 迴歸可能需要 winsorize，但供應商推薦不見得）。

#### Cell 22（markdown）—— histplot 說明

#### Cell 23 — 純直方圖（無 KDE）
- 比 Cell 13 用了更多 bins（100），可以看細節。

#### Cell 24 — 空 cell
- 可刪。

---

### 【區塊 A-7】數值特徵相關性

#### Cell 25 — 列出 df_copy_new 的數值欄位

#### Cell 26 — 相關係數計算
```python
correlations = df[analysis_cols].corr()  # ⚠️ 這裡是 df，不是 df_copy_new
savings_pct_correlations = correlations['Savings Pct'].drop('Savings Pct').sort_values(ascending=False)
```

**⚠️ 重要邏輯錯誤**：
- 變數 `analysis_cols` 來自 `df_copy_new`（洩漏欄位已刪），但 `correlations = df[analysis_cols].corr()` 是用**原始 df**去計算。
- 這行本身雖然不會出錯（因為 df 也有這些欄位），但**沒有納入洩漏欄位的相關性視覺化，是不是刻意跳過確認一次**？
- 建議：應該分別跑一次「含洩漏欄位」和「不含洩漏欄位」的相關矩陣，前者用來**證明洩漏**（給評審看 r ≈ 0.99），後者用來**選特徵**。

---

### 【區塊 A-8】PCA 主成分分析

#### Cell 27（markdown）—— PCA 說明

#### Cell 28 — 執行 PCA
```python
features_for_pca = ['Unit Price', 'Quantity', 'Discount Pct', 'Tax Pct',
                    'Budget Unit Price', 'Savings Pct', 'Days Late',
                    'Lead Time Days', 'Supplier ESG Score', 'Supplier Tier']
x = df[features_for_pca].fillna(df[features_for_pca].median())
x_scaled = StandardScaler().fit_transform(x)
pca = PCA(); pca_features = pca.fit_transform(x_scaled)
```
- 步驟：選特徵 → 中位數填 NaN → 標準化 → PCA → 陡坡圖（Scree Plot）+ 累積解釋變異。

**⚠️ 兩個方法論問題**：
1. **特徵清單包含洩漏欄位**：`Unit Price`、`Budget Unit Price`、`Savings Pct` 這三個高度共線的變數會**支配 PC1**，導致 PC1 本質上是「價格軸」，掩蓋其他有意義的主成分。若是「探索洩漏結構」目的，這樣做 OK；若是「探索非洩漏特徵的結構」，應該把這三個裡的兩個先拿掉。
2. **用中位數填 Savings Pct 的 NaN**：這些 NaN 來自 Cancelled 訂單（551 筆），用中位數填會**製造假訊號**（讓 Cancelled 訂單看起來有中位數水準的節省率）。更誠實的做法是 `dropna()`，或至少報告有多少列是被填補的。

#### Cell 29–30 — PCA Loadings 熱圖
```python
loadings = pd.DataFrame(pca.components_.T, ..., index=features_for_pca)
sns.heatmap(loadings[['PC1', 'PC2', 'PC3']], annot=True, cmap='RdBu_r')
```
- 看每個原始特徵在前 3 個 PC 上的權重，理解每個 PC 的「經濟意義」。

---

### 【區塊 A-9】MCA 多重對應分析

#### Cell 31 — 安裝 prince 套件

#### Cell 32（markdown）—— MCA 說明

#### Cell 33 — MCA 執行與繪圖
```python
selected_cat_cols = ['Category', 'Department', 'Supplier Risk', 'Supplier Status', 'PO Type']
mca_data = df[selected_cat_cols].astype(str)
mca = prince.MCA(n_components=2, ...).fit(mca_data)
column_coords_df = mca.column_coordinates(mca_data)
# 手動用 matplotlib 畫，避開 prince 的 plot_coordinates() 相容性問題
```
- **目的**：把類別變數投影到二維空間，看類別之間的關聯性（例如「Preferred 供應商」是否與「Low Risk」聚在一起）。
- **⚠️ 已知結論**：`Preferred Supplier = Yes` 100% 對應 `Supplier Risk = Low`（memory 已記載），MCA 圖上會清楚看到這兩點重疊。這在報告裡可以直接指出「這兩個變數是冗餘信號」。

---

### 【區塊 B-1】供應商推薦模型：資料重載

#### Cell 34–36（markdown）—— Model 2 說明
- Y = Supplier Name
- X = Savings Pct, On Time Delivery, Supplier Risk, ESG Score, Preferred Supplier
- **⚠️ 方法論警告**（memory 已記載並溝通過）：只有 15 家供應商 → 無 supervised ML 可學。這個區塊應該走 **MCDA（Entropy + TOPSIS）** 而不是 ML。目前 Cell 42/45 做的**其實已經是 MCDA 前置**（多維度供應商績效聚合），只是還沒加權排序。

#### Cell 37 — 重新載入 data.csv
```python
df_original = pd.read_csv(...)
df_sub = df_original.copy()
```
- **關鍵設計**：這裡切斷了 Track A 的 `df`／`df_copy_new` 處理鏈，用「乾淨的原始資料」重新開始 Track B。
- 這樣做的好處：Track A 對 Cancelled 訂單的 NaN 化不會影響 Track B 的決策（因為 Track B 是直接**刪除** Cancelled）。

#### Cell 38 — 刪除 Cancelled 訂單
```python
df_sub = df_sub[df_sub['PO Status'] != 'Cancelled']
# → 4,649 筆
```
- **與 Track A 的差異**：Track A 用 NaN 標記，Track B 直接刪除。
- **⚠️ 對 Subtopic B 的影響**（memory 有記錄）：這個刪除會拿掉 Subtopic B 的正樣本一部分（因為 Cancelled 是二元分類的正類之一）。如果要同時做 Subtopic B，應該另外保留一個含 Cancelled 的分支。

---

### 【區塊 B-2】供應商績效聚合

#### Cell 42 — Aggregation by Category × Supplier Name（mean 版本）

```python
df_eda['On Time Delivery Rate'] = (df_eda['On Time Delivery'] == 'Yes').astype(int)
df_eda['Preferred Flag']       = (df_eda['Preferred Supplier'] == 'Yes').astype(int)
df_eda['Maverick Flag']        = (df_eda['Maverick Spend'] == 'Yes').astype(int)

supplier_performance = df_eda.groupby(['Category', 'Supplier Name']).agg({
    'Savings Pct':           'mean',
    'Savings Amount':        'sum',
    'On Time Delivery Rate': 'mean',
    'Days Late':             'mean',
    'Supplier ESG Score':    'mean',
    'Preferred Flag':        'max',
    'Maverick Flag':         'mean',
    'PO Number':             'count',   # → Transaction_Count
})
```
- **每個聚合函數的邏輯**：
  - `Savings Pct` → mean：該供應商在該類別下的**平均節省率**
  - `Savings Amount` → sum：**總節省金額**（絕對貢獻）
  - `On Time Delivery Rate` → mean：**準時率**（0/1 平均 = 比例）
  - `Days Late` → mean：**平均延遲天數**（負數 = 提早）
  - `Supplier ESG Score` → mean：ESG 分數（memory 記載：供應商內無變異，用 max/mean/min 都一樣）
  - `Preferred Flag` → max：只要曾經是 Preferred 就是 1
  - `Maverick Flag` → mean：**特約外支出比例**
  - `PO Number` → count：**交易筆數**（memory：可作為權重參考）

#### Cell 43 — 視覺化
- **氣泡圖**：X = ESG、Y = Savings Pct、大小 = On-Time Rate、顏色 = Preferred。
- **回歸圖**：Maverick Rate vs Savings Pct。
- **⚠️ 顧問提醒**：
  - `Total_Savings_Amt` 用 `sum` 會被**採購金額規模**主導（例如 IT Software 供應商的絕對節省金額一定比 Office Supplies 大）。這符合 memory 的「Scale confounds vs. genuine signals」原則——這個欄位可以看但**不能當作跨類別比較的公平指標**。
  - 氣泡圖只用了三個維度，之後 MCDA 排序時應該用所有 9 個 KPI 一起。

#### Cell 45 — Aggregation（median 版本）
- 除了改用 `median` 代替 `mean`，結構與 Cell 42 完全相同。
- **⚠️ median 對 binary 變數的問題**：
  - `On Time Delivery Rate` 的值是 0 或 1，取 median 只會得到 0、0.5 或 1（不像 mean 可以是 0.73 這種連續比例）。
  - `Maverick Flag` 同樣問題。
  - **對這兩個欄位而言，mean 才是正確的**（因為它們的語意本來就是「比例」）。
  - 建議：Cell 45 混用——比例類用 mean，連續類（Savings Pct、Days Late、ESG）用 median。

---

## 三、Pipeline 邏輯總評

### 3.1 pipeline 分岔點

```
data.csv (5,200 rows)
    │
    ├─── Track A (df) ──── 對 Cancelled NaN 化 ──── df_copy_new (刪 8 欄位)
    │                                                   │
    │                                                   ├─→ 目標變數 EDA
    │                                                   ├─→ 相關矩陣（有 bug：用 df 不是 df_copy_new）
    │                                                   ├─→ PCA（有洩漏欄位混入）
    │                                                   └─→ MCA
    │
    └─── Track B (df_sub) ── 刪 Cancelled (4,649 rows) ── Supplier × Category 聚合
                                                              │
                                                              ├─→ Cell 42（mean）
                                                              └─→ Cell 45（median）
```

### 3.2 值得指出的邏輯問題（優先順序）

| 級別 | 問題 | 位置 | 建議 |
|---|---|---|---|
| 🔴 高 | Savings Pct = NaN 但 markdown 說「設為 0」 | Cell 3 vs 4 | 修 markdown 為「設為 NaN」以求文字與程式一致 |
| 🔴 高 | Cell 26 相關矩陣使用 `df` 而非 `df_copy_new` | Cell 26 | 用 `df_copy_new` 才符合「刪洩漏欄位後」的分析邏輯 |
| 🔴 高 | PCA 特徵包含高共線洩漏欄位（Unit Price、Budget Unit Price、Savings Pct） | Cell 28 | 洩漏欄位版本與去洩漏版本分開跑，結論才乾淨 |
| 🟡 中 | 類別變數用 LabelEncoder + Spearman | Cell 18 | 名目類別應改用 ANOVA F 或 η² |
| 🟡 中 | median 版本的 On Time Rate / Maverick Rate 語意破壞 | Cell 45 | 比例類仍用 mean、連續類再用 median |
| 🟡 中 | Cell 5 沒同步處理 Days Late | Cell 5 | 若 Days Late 要當特徵，Cancelled 應一致 NaN 化 |
| 🟢 低 | Quantity 被列入刪除清單 | Cell 10 | Subtopic A 建模時保留 Quantity（不是 formula chain） |
| 🟢 低 | Cell 15、Cell 24 空 cell / 冗餘 cell | — | 清理 notebook 前刪掉 |

### 3.3 這個 EDA 為後續建模鋪墊了什麼

| Subtopic | 這份 EDA 提供了什麼 |
|---|---|
| **A（PPV 迴歸）** | 洩漏欄位清單已識別；Savings Pct 分佈與離群值範圍已掌握；類別特徵的重要性排序（雖然方法待改） |
| **B（Cancelled+Disputed 分類）** | ⚠️ **這份 EDA 完全沒觸及**——Cancelled 被當雜訊清掉了。Subtopic B 需要獨立的 EDA notebook |
| **C（供應商推薦 MCDA）** | Cell 42/45 的聚合結果已經是 TOPSIS 的輸入矩陣雛形，接下來加 Entropy weighting → TOPSIS 排序即可 |

---

## 四、下一步建議

1. **修 Cell 3 markdown 的文字**（Savings Pct 是 NaN 不是 0）。
2. **修 Cell 26**：用 `df_copy_new` 計算相關矩陣；同時**保留一份含洩漏欄位的版本**當作報告裡的「洩漏證據圖」。
3. **PCA 拆兩版**（含洩漏 / 不含洩漏），對比 loadings 差異。
4. **Cell 45 混用聚合方式**：比例類仍用 mean、連續類用 median。
5. **決定 Subtopic C 路線**（memory 記錄的 Route A / Route A+B），Cell 42/45 輸出已可直接進 TOPSIS。
6. **Subtopic B 需另開 notebook**，且要**不刪 Cancelled**。
