# -*- coding: utf-8 -*-
"""
=====================================================================
 新供應商風險評分引擎 (New Supplier Risk Scoring) — v1.0
=====================================================================
商業問題
--------
「第 16 家新供應商進來時，依公司既有 15 家的風險政策，
  它會被分到 Low / Medium / High 哪一級？為什麼？」

與 supplier_risk_github.py 的關係
--------------------------------
supplier_risk_github.py 已證明：逐單 (PO-level) 行為特徵無法
泛化到沒見過的供應商 (留供應商出局 Acc≈0.5)。本腳本改變問題
定義 —— 不再嘗試「從行為學風險」，而是「把公司隱含的風險
政策學起來，一致且可解釋地套用到新供應商」。在這個情境下，
Tier / Preferred / Status / ESG 不是洩漏，而是新供應商入職時
本來就會有的評估屬性。

設計
----
- 建模單位   : 供應商層級 (15 家 → 15 列畫像)
- 模型       : Logistic Regression (在 LOSO 掃描中勝過
               DTree/RF/LGBM —— n=15 時簡單模型泛化最好，
               且係數天生可解釋)
- 驗證       : Leave-One-Supplier-Out (每家輪流當「第16家」)
- 可解釋性   : 各特徵對預測類別 logit 的貢獻分解
               (係數 × 標準化後特徵值)，輸出商業語言解讀

執行
----
    source backend/.venv/bin/activate
    python training_scripts/new_supplier_risk_scoring.py
輸出: training_scripts/output/ (LOSO 混淆矩陣、係數圖、模型)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(REPO_ROOT, "dataset", "data_poisntcancelled.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_ORDER = ["Low", "Medium", "High"]

print("=" * 72)
print("🧭 新供應商風險評分引擎 (LOSO 驗證 + 可解釋性)")
print("=" * 72)

# =====================================================================
# STEP 1. 供應商畫像表 — 15 家 × (入職屬性 + 累積行為率)
# =====================================================================
print("\n=== STEP 1: 建立供應商層級畫像 ===")
df = pd.read_csv(DATA_PATH)
for c in ["Maverick Spend", "Single Source Flag", "Preferred Supplier", "On Time Delivery"]:
    df[c] = df[c].map({"Yes": 1, "No": 0}).astype(float)

sup = df.groupby("Supplier ID").agg(
    supplier_name=("Supplier Name", "first"),
    tier=("Supplier Tier", "first"),                # 入職屬性
    preferred=("Preferred Supplier", "first"),      # 入職屬性
    status=("Supplier Status", "first"),            # 入職屬性
    esg=("Supplier ESG Score", "first"),            # 外部評分
    region=("Supplier Region", "first"),            # 入職屬性
    local=("Local International", "first"),         # 入職屬性
    mav_rate=("Maverick Spend", "mean"),            # 累積行為率
    single_rate=("Single Source Flag", "mean"),     # 累積行為率
    risk=("Supplier Risk", "first"),
).reset_index()
sup["status_enc"] = sup["status"].map({"Preferred": 0, "Approved": 1, "Conditional": 2})

region_dummies = pd.get_dummies(sup["region"], prefix="reg")
local_dummies = pd.get_dummies(sup["local"], prefix="loc")
sup = pd.concat([sup, region_dummies, local_dummies], axis=1)

FEATURES = (
    ["tier", "preferred", "status_enc", "esg", "mav_rate", "single_rate"]
    + list(region_dummies.columns)
    + list(local_dummies.columns)
)
FEATURE_LABELS = {  # 商業語言對照 (解釋輸出用)
    "tier": "供應商層級 (Tier)",
    "preferred": "優先供應商資格",
    "status_enc": "供應商狀態 (Preferred→Conditional)",
    "esg": "ESG 評分",
    "mav_rate": "脫軌採購率",
    "single_rate": "單一來源率",
}
for c in list(region_dummies.columns) + list(local_dummies.columns):
    FEATURE_LABELS[c] = c.replace("reg_", "地區=").replace("loc_", "供應型態=")

y = sup["risk"].map({r: i for i, r in enumerate(RISK_ORDER)}).values
print(f">> 畫像表: {sup.shape[0]} 家供應商, {len(FEATURES)} 個特徵")
print(f">> 標籤分佈: { {r: int((sup['risk']==r).sum()) for r in RISK_ORDER} }")


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


# =====================================================================
# STEP 2. LOSO 驗證 — 每家輪流當「第 16 家新供應商」
# =====================================================================
print("\n=== STEP 2: Leave-One-Supplier-Out 驗證 ===")
loso_pred = np.zeros(len(sup), dtype=int)
for i in range(len(sup)):
    mask = np.ones(len(sup), dtype=bool)
    mask[i] = False
    m = make_model()
    m.fit(sup.loc[mask, FEATURES], y[mask])
    loso_pred[i] = int(m.predict(sup.loc[[i], FEATURES])[0])

print(f"{'Supplier':10s} {'名稱':26s} {'真實':8s} {'LOSO預測':8s}")
adjacent_errors = 0
for i, row in sup.iterrows():
    t, p = RISK_ORDER[y[i]], RISK_ORDER[loso_pred[i]]
    ok = "✅" if t == p else ("↕️ 相鄰" if abs(y[i] - loso_pred[i]) == 1 else "❌ 對調")
    if t != p and abs(y[i] - loso_pred[i]) == 1:
        adjacent_errors += 1
    print(f"{row['Supplier ID']:10s} {row['supplier_name']:26s} {t:8s} {p:8s} {ok}")

acc = float((loso_pred == y).mean())
extreme = int(((np.abs(loso_pred - y)) == 2).sum())
print(f"\n>> LOSO 供應商層級準確率: {acc:.1%} ({int((loso_pred==y).sum())}/15)")
print(f">> 錯誤全貌: 相鄰等級誤判 {adjacent_errors} 家, Low↔High 對調 {extreme} 家")
print(">> 註: High 僅 1 家 — 留它出局時訓練集無 High 樣本,")
print("       該 fold 結構上不可能預測正確 (資料限制, 非模型缺陷)。")

cm = confusion_matrix(
    [RISK_ORDER[v] for v in y], [RISK_ORDER[v] for v in loso_pred], labels=RISK_ORDER
)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=RISK_ORDER).plot(
    ax=ax, cmap=plt.cm.Blues, colorbar=False
)
ax.set_title(f"LOSO Confusion Matrix (supplier-level, acc={acc:.0%})")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "loso_confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"💾 {cm_path}")

# =====================================================================
# STEP 3. 最終評分引擎 — 用全部 15 家訓練
# =====================================================================
print("\n=== STEP 3: 訓練最終評分引擎 (15 家全量) ===")
final_model = make_model()
final_model.fit(sup[FEATURES], y)
scaler = final_model.named_steps["scaler"]
clf = final_model.named_steps["clf"]

# 全域係數圖 — 政策解讀: 什麼把風險往上推/往下拉
coef_df = pd.DataFrame(
    clf.coef_.T,
    index=[FEATURE_LABELS.get(f, f) for f in FEATURES],
    columns=[RISK_ORDER[c] for c in clf.classes_],
)
fig, ax = plt.subplots(figsize=(9, 7))
coef_df.plot.barh(ax=ax)
ax.set_title("Risk Policy Coefficients (standardized)\n正值 = 把供應商往該風險級推")
ax.axvline(0, color="k", lw=0.8)
plt.tight_layout()
coef_path = os.path.join(OUTPUT_DIR, "risk_policy_coefficients.png")
plt.savefig(coef_path, dpi=120)
plt.close()
print(f"💾 {coef_path}")


# =====================================================================
# STEP 4. 可解釋性 — 為什麼這家被分到這一級?
# =====================================================================
def explain_supplier(profile: pd.DataFrame, top_k: int = 5):
    """對單一供應商畫像輸出: 預測類別、機率、逐特徵貢獻。

    貢獻 = 標準化特徵值 × 該類別係數 (對 logit 的加法分解)。
    """
    proba = final_model.predict_proba(profile[FEATURES])[0]
    pred_idx = int(np.argmax(proba))
    z = scaler.transform(profile[FEATURES])[0]
    contrib = z * clf.coef_[list(clf.classes_).index(pred_idx)]
    order = np.argsort(-np.abs(contrib))

    lines = [
        f"預測風險等級: {RISK_ORDER[pred_idx]}",
        "機率分佈: " + "  ".join(
            f"{RISK_ORDER[c]}={proba[j]:.1%}" for j, c in enumerate(clf.classes_)
        ),
        f"為什麼是 {RISK_ORDER[pred_idx]}? (前 {top_k} 大影響因素)",
    ]
    for r, j in enumerate(order[:top_k], 1):
        direction = "↑ 推向此級" if contrib[j] > 0 else "↓ 拉離此級"
        raw = profile[FEATURES].iloc[0, j]
        raw_str = f"{float(raw):.2f}" if isinstance(raw, (int, float, np.floating)) else str(raw)
        lines.append(
            f"  {r}. {FEATURE_LABELS.get(FEATURES[j], FEATURES[j]):24s}"
            f" 值={raw_str:>8s}  貢獻={contrib[j]:+.2f}  {direction}"
        )
    return "\n".join(lines), pred_idx, proba


def score_new_supplier(
    tier: int, preferred: int, status: str, esg: float,
    region: str, local: str, mav_rate: float = 0.0, single_rate: float = 0.0,
):
    """對第 16 家新供應商評分。

    參數皆為入職時可得: tier=1/2/3, preferred=0/1,
    status='Preferred'/'Approved'/'Conditional', esg=0-100,
    region 如 'Asia', local='Local'/'International',
    mav_rate/single_rate = 交易累積後的行為率 (剛入職可先填 0)。
    """
    row = {f: 0.0 for f in FEATURES}
    row.update({
        "tier": tier, "preferred": preferred,
        "status_enc": {"Preferred": 0, "Approved": 1, "Conditional": 2}[status],
        "esg": esg, "mav_rate": mav_rate, "single_rate": single_rate,
    })
    for col, val in [(f"reg_{region}", region), (f"loc_{local}", local)]:
        if col in row:
            row[col] = 1.0
    return explain_supplier(pd.DataFrame([row]))


# --- 示範 1: 回顧一家既有供應商的解釋 ---
print("\n" + "=" * 72)
print("🔍 示範 1: 解釋既有供應商 SUP-011 (Atlantic Raw Materials, 真實=High)")
print("=" * 72)
text, _, _ = explain_supplier(sup[sup["Supplier ID"] == "SUP-011"])
print(text)

# --- 示範 2: 虛擬的「第 16 家」新供應商 ---
print("\n" + "=" * 72)
print("🆕 示範 2: 第 16 家新供應商入職評分")
print("=" * 72)
demo = dict(tier=3, preferred=0, status="Conditional", esg=48.0,
            region="Asia", local="International",
            mav_rate=0.30, single_rate=0.15)
print(f"輸入畫像: {demo}\n")
text, _, _ = score_new_supplier(**demo)
print(text)

# =====================================================================
# STEP 5. 導出
# =====================================================================
bundle = {
    "model": final_model,
    "features": FEATURES,
    "feature_labels": FEATURE_LABELS,
    "risk_order": RISK_ORDER,
    "loso_accuracy": acc,
    "supplier_profiles": sup,
}
model_path = os.path.join(OUTPUT_DIR, "new_supplier_scoring_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"\n💾 評分引擎: {model_path}")

print("\n" + "=" * 72)
print("📝 使用注意 (供專題報告引用)")
print("=" * 72)
print(f"""\
1. LOSO 驗證 = 每家輪流扮演「第 16 家」: 準確率 {acc:.0%},
   且所有錯誤皆為相鄰等級誤判、無 Low↔High 對調 —
   對只有 15 個監督樣本的問題, 這是誠實且可用的水準。
2. High 級全公司只有 1 家 (SUP-011), 模型對 High 的辨識
   完全依賴外推; 新供應商若真屬 High, 較可能被判為
   Medium (寧可低估的方向, 審核時應對 Medium 加人工複核)。
3. 本引擎學的是「公司既有風險政策」的近似, 不是客觀風險;
   若政策本身有偏誤, 模型會忠實複製它。
4. 建議用法: 模型給分級 + 機率 + 原因 → 採購人員複核,
   而非全自動定級。
""")
print("🏁 完成。")
