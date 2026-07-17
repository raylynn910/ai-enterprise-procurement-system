# -*- coding: utf-8 -*-
"""
=====================================================================
 新供應商風險評分引擎 (New Supplier Risk Scoring) — v2.0 審計修正版
=====================================================================
商業問題
--------
「第 16 家新供應商進來時，依公司既有 15 家的風險政策，
  它會被分到 Low / Medium / High 哪一級？為什麼？」

v2.0 相對 v1.0 的審計修正 (每一項都經數據驗證)
--------------------------------------------
[漏洞1] train/serve 不一致: v1 的 LOSO 讓被留出的供應商帶著
    「全歷史」行為率受測 —— 但真正的新供應商在入職當下沒有
    任何交易歷史。v2 拆成兩個誠實模式:
      - Day-0 模式  : 只用入職屬性 (tier, esg)
      - Review 模式 : 屬性 + 「目前為止觀察到」的行為率;
        LOSO 驗證時被留出者只拿它「前 k 筆」交易的行為率
[漏洞2] 指紋特徵: region/local 在 15 家中近乎供應商身分代理
    (Americas 僅 2 家、Oceania 僅 1 家)，「Americas→High」是從
    單一樣本學到的偽規則 → v2 全數移除
[漏洞3] 完全共線: Tier ⇔ Supplier Status ⇔ Preferred Supplier
    是同一變數的三種寫法 (Tier1=Preferred=優先, Tier2=Approved,
    Tier3=Conditional)，解釋輸出會三重計同一訊號 → 只保留 tier
[漏洞4] 選擇偏誤: v1 的 73% 是在 16 種「特徵×模型」組合中
    用同一份 LOSO 挑出的最大值，屬樂觀估計 → v2 固定單一設計
    (LogReg)，數字直接報告、不再挑選

誠實結果 (LOSO, 每輪只用 14 家訓練、預測被留出的 1 家)
------------------------------------------------------
    Day-0  模式:               10/15 (66.7%), 零 Low↔High 對調
    Review 模式 (前50筆交易後): 11/15 (73.3%), 零 Low↔High 對調

執行
----
    source backend/.venv/bin/activate
    python training_scripts/new_supplier_risk_scoring.py
輸出: training_scripts/output/ (LOSO 混淆矩陣、係數圖、模型包)
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
REVIEW_K = 50  # Review 模式 LOSO 給被留出者的「前 k 筆」交易數

print("=" * 72)
print("🧭 新供應商風險評分引擎 v2.0 (審計修正版)")
print("=" * 72)

# =====================================================================
# STEP 1. 供應商畫像 — 只保留「非指紋、非共線」的政策屬性與行為率
# =====================================================================
print("\n=== STEP 1: 建立供應商層級畫像 ===")
df = pd.read_csv(DATA_PATH)
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df = df.sort_values("PO Date").reset_index(drop=True)
for c in ["Maverick Spend", "Single Source Flag"]:
    df[c] = df[c].map({"Yes": 1, "No": 0}).astype(float)

sup = df.groupby("Supplier ID").agg(
    supplier_name=("Supplier Name", "first"),
    tier=("Supplier Tier", "first"),          # 入職屬性 (status/preferred 與其完全共線, 不重複計)
    esg=("Supplier ESG Score", "first"),      # 外部評分
    mav_rate=("Maverick Spend", "mean"),      # 全歷史行為率 (既有供應商可得)
    single_rate=("Single Source Flag", "mean"),
    risk=("Supplier Risk", "first"),
).reset_index()

FEATURES_DAY0 = ["tier", "esg"]
FEATURES_REVIEW = ["tier", "esg", "mav_rate", "single_rate"]
FEATURE_LABELS = {
    "tier": "供應商層級 (Tier)",
    "esg": "ESG 評分",
    "mav_rate": "脫軌採購率 (觀察至今)",
    "single_rate": "單一來源率 (觀察至今)",
}

y = sup["risk"].map({r: i for i, r in enumerate(RISK_ORDER)}).values
print(f">> 畫像表: {len(sup)} 家 | Day-0 特徵: {FEATURES_DAY0} | Review 特徵: {FEATURES_REVIEW}")
print(f">> 標籤分佈: { {r: int((sup['risk']==r).sum()) for r in RISK_ORDER} }")
print(">> 已排除: region/local (指紋), status/preferred (與 tier 完全共線)")


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def early_rates(sid: str, k: int):
    """該供應商「前 k 筆」交易的行為率 — 模擬新供應商交易初期。"""
    s = df[df["Supplier ID"] == sid].head(k)
    return float(s["Maverick Spend"].mean()), float(s["Single Source Flag"].mean())


# =====================================================================
# STEP 2. 忠實 LOSO — 被留出者只能用「該時點可得」的資訊
# =====================================================================
print("\n=== STEP 2: LOSO 驗證 (每輪 14 家訓練 → 預測被留出的 1 家) ===")


def loso(mode: str, k: int | None = None):
    feats = FEATURES_DAY0 if mode == "day0" else FEATURES_REVIEW
    preds = np.zeros(len(sup), dtype=int)
    for i, sid in enumerate(sup["Supplier ID"]):
        mask = np.ones(len(sup), dtype=bool)
        mask[i] = False
        m = make_model()
        m.fit(sup.loc[mask, feats], y[mask])  # 訓練: 既有 14 家 (全歷史率)
        row = sup.loc[[i], feats].copy()
        if mode == "review":                   # 測試: 新供應商只有前 k 筆
            mv, sg = early_rates(sid, k)
            row["mav_rate"], row["single_rate"] = mv, sg
        preds[i] = int(m.predict(row)[0])
    return preds


results = {}
for tag, mode, k in [("Day-0 入職", "day0", None),
                     (f"Review (前{REVIEW_K}筆後)", "review", REVIEW_K)]:
    preds = loso(mode, k)
    acc = float((preds == y).mean())
    adj = int(sum(1 for i in range(len(y)) if preds[i] != y[i] and abs(preds[i] - y[i]) == 1))
    ext = int(sum(1 for i in range(len(y)) if abs(preds[i] - y[i]) == 2))
    results[mode] = {"preds": preds, "acc": acc, "adjacent": adj, "extreme": ext}
    print(f"\n--- {tag}: {int((preds==y).sum())}/15 ({acc:.1%}) | 相鄰錯 {adj} | Low↔High 對調 {ext} ---")
    for i, r in sup.iterrows():
        t, p = RISK_ORDER[y[i]], RISK_ORDER[preds[i]]
        mark = "✅" if t == p else ("↕️ 相鄰" if abs(y[i] - preds[i]) == 1 else "❌ 對調")
        print(f"  {r['Supplier ID']}  {r['supplier_name']:26s} 真實={t:7s} 預測={p:7s} {mark}")

print("\n>> 註1: High 全公司僅 1 家 — 留它出局時訓練集無 High 樣本,")
print("        該 fold 結構上不可能預測正確 (資料限制, 非模型缺陷)。")
print(">> 註2: 本設計 (LogReg + 這組特徵) 為事前固定, 未用本 LOSO 挑模型,")
print("        避免選擇偏誤; 早期探索中的 73% 含指紋特徵, 已棄用。")

# LOSO 混淆矩陣 (兩模式並列)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (mode, tag) in zip(axes, [("day0", "Day-0"), ("review", f"Review k={REVIEW_K}")]):
    cm = confusion_matrix([RISK_ORDER[v] for v in y],
                          [RISK_ORDER[v] for v in results[mode]["preds"]], labels=RISK_ORDER)
    ConfusionMatrixDisplay(cm, display_labels=RISK_ORDER).plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(f"LOSO — {tag} (acc={results[mode]['acc']:.0%})")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "loso_confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"💾 {cm_path}")

# =====================================================================
# STEP 3. 最終評分引擎 — 兩個模式各訓練一顆 (15 家全量)
# =====================================================================
print("\n=== STEP 3: 訓練最終評分引擎 (15 家全量, day0 + review 各一) ===")
day0_model = make_model().fit(sup[FEATURES_DAY0], y)
review_model = make_model().fit(sup[FEATURES_REVIEW], y)

# 政策係數圖 (review 模式)
clf_r = review_model.named_steps["clf"]
coef_df = pd.DataFrame(
    clf_r.coef_.T,
    index=[FEATURE_LABELS[f] for f in FEATURES_REVIEW],
    columns=[RISK_ORDER[c] for c in clf_r.classes_],
)
fig, ax = plt.subplots(figsize=(8, 5))
coef_df.plot.barh(ax=ax)
ax.set_title("Risk Policy Coefficients (standardized, review mode)\n正值 = 把供應商往該風險級推")
ax.axvline(0, color="k", lw=0.8)
plt.tight_layout()
coef_path = os.path.join(OUTPUT_DIR, "risk_policy_coefficients.png")
plt.savefig(coef_path, dpi=120)
plt.close()
print(f"💾 {coef_path}")


# =====================================================================
# STEP 4. 可解釋性
# =====================================================================
def explain(profile: pd.DataFrame, mode: str, top_k: int = 4):
    """輸出: 預測等級、機率、逐特徵貢獻 (係數 × 標準化特徵值)。"""
    model = day0_model if mode == "day0" else review_model
    feats = FEATURES_DAY0 if mode == "day0" else FEATURES_REVIEW
    proba = model.predict_proba(profile[feats])[0]
    clf = model.named_steps["clf"]
    pred_idx = int(np.argmax(proba))
    z = model.named_steps["scaler"].transform(profile[feats])[0]
    contrib = z * clf.coef_[list(clf.classes_).index(pred_idx)]
    order = np.argsort(-np.abs(contrib))

    lines = [
        f"模式: {'Day-0 入職' if mode=='day0' else 'Review (交易累積後)'}",
        f"預測風險等級: {RISK_ORDER[pred_idx]}",
        "機率分佈: " + "  ".join(
            f"{RISK_ORDER[c]}={proba[j]:.1%}" for j, c in enumerate(clf.classes_)),
        f"為什麼是 {RISK_ORDER[pred_idx]}?",
    ]
    for r, j in enumerate(order[:top_k], 1):
        direction = "↑ 推向此級" if contrib[j] > 0 else "↓ 拉離此級"
        lines.append(
            f"  {r}. {FEATURE_LABELS[feats[j]]:22s} 值={float(profile[feats].iloc[0, j]):8.2f}"
            f"  貢獻={contrib[j]:+.2f}  {direction}")
    return "\n".join(lines)


def score_new_supplier(tier: int, esg: float,
                       mav_rate: float | None = None,
                       single_rate: float | None = None,
                       n_transactions: int | None = None):
    """對新供應商評分。

    入職當下 (無交易歷史): 只給 tier + esg → Day-0 模式。
    交易一段時間後: 加上觀察到的 mav_rate / single_rate → Review 模式;
    n_transactions < 25 時行為率統計仍不穩, 會提醒以 Day-0 為主。
    """
    if mav_rate is None or single_rate is None:
        return explain(pd.DataFrame([{"tier": tier, "esg": esg}]), "day0")
    text = explain(pd.DataFrame([{
        "tier": tier, "esg": esg, "mav_rate": mav_rate, "single_rate": single_rate
    }]), "review")
    if n_transactions is not None and n_transactions < 25:
        text += f"\n  ⚠️ 交易僅 {n_transactions} 筆, 行為率仍不穩定, 建議並看 Day-0 結果。"
    return text


print("\n" + "=" * 72)
print("🆕 示範 1: 第 16 家新供應商 — 入職當下 (Day-0)")
print("=" * 72)
print(score_new_supplier(tier=3, esg=48.0))

print("\n" + "=" * 72)
print("🔄 示範 2: 同一家 — 交易 60 筆後 (Review)")
print("=" * 72)
print(score_new_supplier(tier=3, esg=48.0, mav_rate=0.55, single_rate=0.10,
                         n_transactions=60))

# =====================================================================
# STEP 5. 導出
# =====================================================================
bundle = {
    "day0_model": day0_model,
    "review_model": review_model,
    "features_day0": FEATURES_DAY0,
    "features_review": FEATURES_REVIEW,
    "feature_labels": FEATURE_LABELS,
    "risk_order": RISK_ORDER,
    "loso_accuracy": {"day0": results["day0"]["acc"], "review": results["review"]["acc"]},
    "review_k": REVIEW_K,
}
model_path = os.path.join(OUTPUT_DIR, "new_supplier_scoring_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"\n💾 評分引擎: {model_path}")

print("\n" + "=" * 72)
print("📝 使用注意 (供專題報告引用)")
print("=" * 72)
print(f"""\
1. LOSO 為驗證協定 (15 顆臨時模型用後即棄); 最終部署的兩顆模型
   (day0 / review) 皆以全部 15 家重新訓練。
2. 誠實泛化估計: Day-0 {results['day0']['acc']:.0%} → Review(前{REVIEW_K}筆後)
   {results['review']['acc']:.0%}; 兩模式錯誤皆為相鄰等級, 無 Low↔High 對調。
3. High 級僅 1 家: 真實 High 的新供應商較可能被判 Medium
   (系統性低估) → Medium 以上一律人工複核。
4. n=15 的機率輸出未經校準, 98% 不代表真的 98% — 只看分級與
   排序, 別把機率當信賴度。
5. 本引擎學的是「公司既有風險政策」的近似; 政策有偏, 模型照抄。
""")
print("🏁 完成。")
