# -*- coding: utf-8 -*-
"""
=====================================================================
 新供應商風險評分引擎 (New Supplier Risk Scoring) — v3.0 二元框架
=====================================================================
主框架:  Low → 「核准」(自動核准)  vs  Medium+High → 「需複核」(人工複核)

商業問題
--------
「第 16 家新供應商進來時，依公司既有風險政策，它該『自動核准』
  還是『送人工複核』？為什麼？」

為什麼從三分類(Low/Medium/High)改成二元(經對抗式雙 agent 驗證)
--------------------------------------------------------------
1. 三分類的 High 全公司僅 1 家(SUP-011)→ LOSO 留它出局時訓練集無
   High → 該折結構上必錯，硬天花板 14/15。二元併類後此「結構性 High
   枷鎖」消失，每折訓練集恆有正類。
2. 二元「核准 vs 人工複核」恰好對齊真正的營運決策。
3. **誠實提醒**(供報告引用，不可省略):
   - 二元帳面較高 **不代表判別力增強**。相對三分類 tier 查表 73.3%→80%
     的提升，幾乎全來自唯一 High(SUP-011)併類後不再受結構性懲罰的
     *機械效果*，非新學到的判別力。
   - **禁止**把二元 86.7% 與三分類 73.3% 直接對比(不同任務；二元有
     60% 多數類白送基準)。
   - 核心難點 Tier2 的 Low↔中高不可分(SUP-009 漏放、SUP-014 誤殺)
     **原封未動**。

協定說明(避免 v2 的 train/serve 不一致)
------------------------------------------
Review 模式的行為率，訓練列與部署/服務時皆採「前 k 筆」——既有供應商
與新供應商站在同一基準，消除 v2「訓練吃全歷史、服務吃早期率」的偏移。

執行
----
    source backend/.venv/bin/activate
    python training_scripts/new_supplier_risk_scoring.py
輸出: training_scripts/output/ + 自動部署至 backend/models/
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
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(REPO_ROOT, "dataset", "data_poisntcancelled.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 二元標籤: 0 = 核准 (原 Low) | 1 = 需複核 (原 Medium+High)
BINARY_LABELS = ["核准", "需複核"]
REVIEW_K = 50       # Review 模式行為率的「前 k 筆」交易數
N_PERM = 1000       # permutation 檢定次數

print("=" * 72)
print("🧭 新供應商風險評分引擎 v3.0 (二元框架: 核准 / 需複核)")
print("=" * 72)

# =====================================================================
# STEP 1. 供應商畫像 — 非指紋、非共線特徵 + as-of 行為率
# =====================================================================
print("\n=== STEP 1: 建立供應商層級畫像 (二元標籤) ===")
df = pd.read_csv(DATA_PATH)
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df = df.sort_values("PO Date").reset_index(drop=True)
for c in ["Maverick Spend", "Single Source Flag"]:
    df[c] = df[c].map({"Yes": 1, "No": 0}).astype(float)

sup = df.groupby("Supplier ID").agg(
    supplier_name=("Supplier Name", "first"),
    tier=("Supplier Tier", "first"),
    esg=("Supplier ESG Score", "first"),
    mav_rate=("Maverick Spend", "mean"),        # 全歷史率(Day-0 不用；僅備查)
    single_rate=("Single Source Flag", "mean"),
    risk=("Supplier Risk", "first"),
).reset_index()

# 二元標籤: Low → 0 (核准), 其餘 → 1 (需複核)
y = (sup["risk"] != "Low").astype(int).values

# Review 用「前 k 筆」行為率 — 訓練與服務同基準(對稱協定)
def early_rates(sid: str, k: int):
    s = df[df["Supplier ID"] == sid].head(k)
    return float(s["Maverick Spend"].mean()), float(s["Single Source Flag"].mean())

_early = {sid: early_rates(sid, REVIEW_K) for sid in sup["Supplier ID"]}
sup_rev = sup.copy()
sup_rev["mav_rate"] = sup_rev["Supplier ID"].map(lambda s: _early[s][0])
sup_rev["single_rate"] = sup_rev["Supplier ID"].map(lambda s: _early[s][1])

FEATURES_DAY0 = ["tier", "esg"]
FEATURES_REVIEW = ["tier", "esg", "mav_rate", "single_rate"]
FEATURE_LABELS = {
    "tier": "供應商層級 (Tier)",
    "esg": "ESG 評分",
    "mav_rate": "脫軌採購率 (觀察至今)",
    "single_rate": "單一來源率 (觀察至今)",
}
SUP_TABLE = {"day0": sup, "review": sup_rev}   # 各模式對應的畫像表

n_pos = int(y.sum())
majority_acc = max(n_pos, len(y) - n_pos) / len(y)
print(f">> 畫像表: {len(sup)} 家 | 核准(0)={len(y)-n_pos} 家, 需複核(1)={n_pos} 家")
print(f">> 多數類基準(全猜核准) = {len(y)-n_pos}/{len(y)} = {majority_acc:.1%}")
print(">> 已排除: region/local (指紋), status/preferred (與 tier 完全共線)")


def make_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def _pos_class_contrib(clf, z):
    """回傳 z 對『需複核(正類=1)』logit 的逐特徵貢獻。

    二元 LogisticRegression 的 coef_ 形狀為 (1, n_features)，該列即為
    正類方向；不可用 coef_[pred_pos]（pred_pos=1 會 IndexError）。
    """
    return z * clf.coef_[0]


# =====================================================================
# STEP 2. 忠實 LOSO — 每輪 14 家訓練、預測留出 1 家 (對稱前 k 協定)
# =====================================================================
print("\n=== STEP 2: LOSO 驗證 (二元, group-out) ===")


def loso(mode: str):
    tbl = SUP_TABLE[mode]
    feats = FEATURES_DAY0 if mode == "day0" else FEATURES_REVIEW
    preds = np.zeros(len(tbl), dtype=int)
    proba1 = np.zeros(len(tbl))            # P(需複核)
    for i in range(len(tbl)):
        mask = np.ones(len(tbl), dtype=bool)
        mask[i] = False
        m = make_model()
        m.fit(tbl.loc[mask, feats], y[mask])
        clf = m.named_steps["clf"]
        pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else None
        xrow = tbl.loc[[i], feats]
        preds[i] = int(m.predict(xrow)[0])
        proba1[i] = float(m.predict_proba(xrow)[0][pos_col]) if pos_col is not None else 0.0
    return preds, proba1


def _perm_pvalue(mode: str, real_acc: float):
    """打亂標籤重跑 LOSO 的 permutation p 值 (acc)。"""
    tbl = SUP_TABLE[mode]
    feats = FEATURES_DAY0 if mode == "day0" else FEATURES_REVIEW
    ge = 0
    for _ in range(N_PERM):
        yp = rng.permutation(y)
        correct = 0
        for i in range(len(tbl)):
            mask = np.ones(len(tbl), dtype=bool); mask[i] = False
            if len(np.unique(yp[mask])) < 2:
                continue
            m = make_model(); m.fit(tbl.loc[mask, feats], yp[mask])
            correct += int(m.predict(tbl.loc[[i], feats])[0] == yp[i])
        if correct / len(tbl) >= real_acc:
            ge += 1
    return (ge + 1) / (N_PERM + 1)


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z*z/n
    c = p + z*z/(2*n)
    m = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (c - m)/d, (c + m)/d


results = {}
for tag, mode in [("Day-0 入職", "day0"), (f"Review (前{REVIEW_K}筆)", "review")]:
    preds, proba1 = loso(mode)
    acc = float((preds == y).mean())
    auc = float(roc_auc_score(y, proba1))
    mf1 = float(f1_score(y, preds, average="macro"))
    pr = precision_recall_fscore_support(y, preds, labels=[0, 1], zero_division=0)
    lo, hi = _wilson(int((preds == y).sum()), len(y))
    results[mode] = {"preds": preds, "proba1": proba1, "acc": acc, "auc": auc,
                     "macro_f1": mf1, "wilson": (lo, hi)}
    print(f"\n--- {tag}: {int((preds==y).sum())}/15 ({acc:.1%}) | "
          f"AUC={auc:.2f} | macroF1={mf1:.2f} | Wilson95%=[{lo:.0%},{hi:.0%}] "
          f"(基準 {majority_acc:.0%}) ---")
    print(f"    核准  P/R = {pr[0][0]:.2f}/{pr[1][0]:.2f} | "
          f"需複核 P/R = {pr[0][1]:.2f}/{pr[1][1]:.2f}")
    for i, r in sup.iterrows():
        t, p = BINARY_LABELS[y[i]], BINARY_LABELS[preds[i]]
        mark = "✅" if t == p else "❌"
        print(f"  {r['Supplier ID']}  {r['supplier_name']:26s} "
              f"真實={t:4s} 預測={p:4s} {mark}")

print("\n>> permutation 檢定 (判別力是否超出雜訊)...")
for mode in ["day0", "review"]:
    p = _perm_pvalue(mode, results[mode]["acc"])
    results[mode]["perm_p"] = p
    print(f"   {mode:6s}: acc={results[mode]['acc']:.1%}  permutation p={p:.3f}")

print("\n>> 註1: Tier2 的 Low↔中高不可分性未消 (SUP-009 漏放、SUP-014 誤殺),")
print("        與三分類同一批矛盾樣本; 行為率(Review)相對 Day-0 零貢獻。")
print(">> 註2: 二元帳面高於三分類主要來自唯一 High 併類的機械效果,")
print("        非判別力增強; 不可與三分類 accuracy 直接對比。")

# LOSO 混淆矩陣 (兩模式並列)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (mode, tag) in zip(axes, [("day0", "Day-0"), ("review", f"Review k={REVIEW_K}")]):
    cm = confusion_matrix([BINARY_LABELS[v] for v in y],
                          [BINARY_LABELS[v] for v in results[mode]["preds"]],
                          labels=BINARY_LABELS)
    ConfusionMatrixDisplay(cm, display_labels=BINARY_LABELS).plot(
        ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title(f"LOSO — {tag} (acc={results[mode]['acc']:.0%}, AUC={results[mode]['auc']:.2f})")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "loso_confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"💾 {cm_path}")

# =====================================================================
# STEP 3. 最終評分引擎 — 兩模式各訓練一顆 (15 家全量)
# =====================================================================
print("\n=== STEP 3: 訓練最終評分引擎 (15 家全量) ===")
day0_model = make_model().fit(sup[FEATURES_DAY0], y)
review_model = make_model().fit(sup_rev[FEATURES_REVIEW], y)

# 政策係數圖 (review 模式；二元 → 單列 = 推向『需複核』方向)
clf_r = review_model.named_steps["clf"]
coef_s = pd.Series(clf_r.coef_[0], index=[FEATURE_LABELS[f] for f in FEATURES_REVIEW])
fig, ax = plt.subplots(figsize=(7, 4))
coef_s.sort_values().plot.barh(ax=ax, color=["#2563eb" if v < 0 else "#dc2626" for v in coef_s.sort_values()])
ax.set_title("政策係數 (標準化, 正值=推向『需複核』)")
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
    """輸出: 預測(核准/需複核)、機率、逐特徵貢獻。"""
    model = day0_model if mode == "day0" else review_model
    feats = FEATURES_DAY0 if mode == "day0" else FEATURES_REVIEW
    clf = model.named_steps["clf"]
    proba = model.predict_proba(profile[feats])[0]
    pos_col = list(clf.classes_).index(1)
    p_review = float(proba[pos_col])
    pred = 1 if p_review >= 0.5 else 0

    z = model.named_steps["scaler"].transform(profile[feats])[0]
    contrib_pos = _pos_class_contrib(clf, z)          # 對『需複核』的貢獻
    contrib = contrib_pos if pred == 1 else -contrib_pos   # 對『預測類別』的貢獻
    order = np.argsort(-np.abs(contrib))

    lines = [
        f"模式: {'Day-0 入職' if mode=='day0' else 'Review (交易累積後)'}",
        f"預測: {BINARY_LABELS[pred]}",
        f"機率: 核准={1-p_review:.1%}  需複核={p_review:.1%}",
        f"為什麼是「{BINARY_LABELS[pred]}」?",
    ]
    for r, j in enumerate(order[:top_k], 1):
        direction = "↑ 推向此判定" if contrib[j] > 0 else "↓ 拉離此判定"
        lines.append(
            f"  {r}. {FEATURE_LABELS[feats[j]]:22s} 值={float(profile[feats].iloc[0, j]):8.2f}"
            f"  貢獻={contrib[j]:+.2f}  {direction}")
    return "\n".join(lines)


def score_new_supplier(tier: int, esg: float,
                       mav_rate: float | None = None,
                       single_rate: float | None = None,
                       n_transactions: int | None = None):
    """對新供應商評分。行為率請帶入『觀察至今前 k 筆』的比率(與訓練同基準)。"""
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
print(score_new_supplier(tier=3, esg=48.0, mav_rate=0.55, single_rate=0.10, n_transactions=60))

# =====================================================================
# STEP 5. 導出 + 自動部署
# =====================================================================
bundle = {
    "task": "binary",
    "day0_model": day0_model,
    "review_model": review_model,
    "features_day0": FEATURES_DAY0,
    "features_review": FEATURES_REVIEW,
    "feature_labels": FEATURE_LABELS,
    "risk_order": BINARY_LABELS,          # index by label 0/1
    "loso_accuracy": {"day0": results["day0"]["acc"], "review": results["review"]["acc"]},
    "loso_auc": {"day0": results["day0"]["auc"], "review": results["review"]["auc"]},
    "majority_baseline": majority_acc,
    "review_k": REVIEW_K,
}
model_path = os.path.join(OUTPUT_DIR, "new_supplier_scoring_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(bundle, f)
BACKEND_MODELS = os.path.join(REPO_ROOT, "backend", "models")
os.makedirs(BACKEND_MODELS, exist_ok=True)
backend_path = os.path.join(BACKEND_MODELS, "new_supplier_scoring_model.pkl")
with open(backend_path, "wb") as f:
    pickle.dump(bundle, f)
print(f"\n💾 評分引擎: {model_path}")
print(f"💾 已部署至: {backend_path}")

print("\n" + "=" * 72)
print("📝 使用注意 (供專題報告引用)")
print("=" * 72)
print(f"""\
1. 主框架為二元「核准(Low) vs 需複核(Medium 以上)」, 對齊真實營運決策。
   忠實 LOSO: Day-0 {results['day0']['acc']:.0%} / Review {results['review']['acc']:.0%},
   AUC {results['day0']['auc']:.2f}, 顯著優於多數類基準 {majority_acc:.0%}
   (permutation p={results['day0']['perm_p']:.3f})。
2. 誠實邊界: 二元帳面高於三分類, 主要來自唯一 High 併類的機械效果,
   非判別力增強; 且二元有 {majority_acc:.0%} 白送基準, 不可與三分類 accuracy 直接比。
3. 本引擎學的是「公司既有風險政策」的近似; 政策有偏, 模型照抄。
   若有風險疑慮，一律走人工複核。
""")
print("🏁 完成。")
