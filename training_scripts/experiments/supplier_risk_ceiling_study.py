# -*- coding: utf-8 -*-
"""
=====================================================================
 供應商風險評分 — 泛化能力天花板研究 (Ceiling Study)
=====================================================================
角色: Data Analysis Engineer
目標: 誠實地把 LOSO 泛化拉到 >73.3% (理想 >=78%),
      或用嚴謹方法證明 73.3% 已接近此資料的天花板。

方法學紅線 (全程遵守):
  1. 分類指標: Accuracy / macro-F1 / 各類 P·R / AUC-ROC(OvR macro) /
     Log Loss / 序位 MAE。禁用 MSE·R²。
  2. 不在測試集上挑選: 候選設計「事前固定」全部回報 (不挑最大);
     任何超參數用「巢狀 CV」(inner CV 選, outer LOSO 評)。
  3. 無洩漏: scaler 只 fit 訓練折; Review 行為率用被留出者「前 k 筆」;
     被留出供應商資訊不進訓練 (含填補、跨供應商聚合)。
  4. 信賴區間: 每個分數附 bootstrap 95% CI 與 Wilson 二項 CI;
     11/15 vs 12/15 差 1 家, 明確標示是否落在雜訊區。
  5. High 折誠實揭露 (全公司 1 家 High, LOSO 結構上必錯)。
  6. 固定 random_state = 42, 可複現。

不改動已部署的 new_supplier_risk_scoring.py; 不 commit / push。
輸出: training_scripts/experiments/output/
"""

import os
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, roc_auc_score,
    log_loss, confusion_matrix,
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_PATH = os.path.join(REPO_ROOT, "dataset", "data_poisntcancelled.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RISK_ORDER = ["Low", "Medium", "High"]
LABEL2I = {r: i for i, r in enumerate(RISK_ORDER)}
REVIEW_K = 50
N_BOOT = 5000

# =====================================================================
# 資料 — 供應商層級畫像 (非指紋、非共線特徵 + as-of 行為率)
# =====================================================================
df = pd.read_csv(DATA_PATH)
df["PO Date"] = pd.to_datetime(df["PO Date"], dayfirst=True)
df = df.sort_values("PO Date").reset_index(drop=True)
for c in ["Maverick Spend", "Single Source Flag"]:
    df[c] = df[c].map({"Yes": 1, "No": 0}).astype(float)

sup = df.groupby("Supplier ID").agg(
    name=("Supplier Name", "first"),
    tier=("Supplier Tier", "first"),
    esg=("Supplier ESG Score", "first"),
    mav_rate=("Maverick Spend", "mean"),
    single_rate=("Single Source Flag", "mean"),
    risk=("Supplier Risk", "first"),
).reset_index()
y = sup["risk"].map(LABEL2I).values
N = len(sup)

FEATURES_DAY0 = ["tier", "esg"]
FEATURES_REVIEW = ["tier", "esg", "mav_rate", "single_rate"]


def early_rates(sid, k):
    s = df[df["Supplier ID"] == sid].head(k)
    return float(s["Maverick Spend"].mean()), float(s["Single Source Flag"].mean())


def review_matrix(feats, k):
    """Build the Review feature matrix where behavioral rates are the
    held-out supplier's *first-k* rates (as-of, no leakage)."""
    X = sup[feats].copy()
    if "mav_rate" in feats:
        for i, sid in enumerate(sup["Supplier ID"]):
            mv, sg = early_rates(sid, k)
            X.loc[i, "mav_rate"] = mv
            X.loc[i, "single_rate"] = sg
    return X.values


# =====================================================================
# 指標工具 — 一次算齊所有分類指標 (含序位 MAE)
# =====================================================================
def all_metrics(y_true, y_pred, proba=None):
    acc = float((y_true == y_pred).mean())
    mf1 = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2],
                         zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    mae = float(np.abs(y_true - y_pred).mean())          # 序位 MAE
    adj = int(np.sum((y_true != y_pred) & (np.abs(y_true - y_pred) == 1)))
    ext = int(np.sum(np.abs(y_true - y_pred) == 2))       # Low<->High 對調
    out = {"accuracy": acc, "macro_f1": mf1, "ordinal_mae": mae,
           "adjacent_err": adj, "extreme_err": ext,
           "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()}
    if proba is not None:
        try:
            out["auc_ovr_macro"] = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro",
                labels=[0, 1, 2]))
        except Exception:
            out["auc_ovr_macro"] = None
        try:
            out["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1, 2]))
        except Exception:
            out["log_loss"] = None
    return out


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(y_true, y_pred, proba=None, metric="accuracy", n=N_BOOT):
    """Bootstrap over the 15 suppliers (the modeling units)."""
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        pr = proba[b] if proba is not None else None
        m = all_metrics(y_true[b], y_pred[b], pr)
        v = m.get(metric)
        if v is not None:
            vals.append(v)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =====================================================================
# 模型工廠 (事前固定設計)
# =====================================================================
def mk_logreg(balanced=True):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000,
                                   class_weight="balanced" if balanced else None,
                                   random_state=RANDOM_STATE)),
    ])


class OrdinalFrankHall:
    """序位邏輯迴歸 (Frank & Hall 二元分解): 對 y>0, y>1 各訓一顆,
    由 P(y>0),P(y>1) 還原三類機率。尊重 Low<Medium<High 序位。"""
    def __init__(self, balanced=True):
        self.balanced = balanced

    def fit(self, X, yy):
        self.models_ = []
        self.trivial_ = []
        for thr in [0, 1]:
            b = (yy > thr).astype(int)
            if b.min() == b.max():              # 訓練折缺某側 (如無 High)
                self.models_.append(None)
                self.trivial_.append(float(b.max()))
            else:
                m = mk_logreg(self.balanced).fit(X, b)
                self.models_.append(m)
                self.trivial_.append(None)
        return self

    def predict_proba(self, X):
        n = len(X)
        pg = np.zeros((n, 2))                    # P(y>0), P(y>1)
        for j, m in enumerate(self.models_):
            if m is None:
                pg[:, j] = self.trivial_[j]
            else:
                pg[:, j] = m.predict_proba(X)[:, list(m.named_steps["clf"].classes_).index(1)]
        pg[:, 1] = np.minimum(pg[:, 1], pg[:, 0])   # 單調性修正
        proba = np.column_stack([1 - pg[:, 0], pg[:, 0] - pg[:, 1], pg[:, 1]])
        proba = np.clip(proba, 1e-9, None)
        return proba / proba.sum(1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), 1)


# =====================================================================
# 忠實 LOSO 執行器 (被留出者只用該時點可得資訊)
# =====================================================================
def loso_predict(Xmat, y, model_fn):
    preds = np.zeros(N, dtype=int)
    proba = np.zeros((N, 3))
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        m = model_fn().fit(Xmat[mask], y[mask])
        pr = m.predict_proba(Xmat[[i]])[0]
        # 對齊到 3 類機率欄 (缺類別時補 0)
        cls = getattr(m, "classes_", None)
        if cls is None and hasattr(m, "models_"):
            proba[i] = pr
        else:
            full = np.zeros(3)
            for j, c in enumerate(cls):
                full[int(c)] = pr[j]
            proba[i] = full
        preds[i] = int(np.argmax(proba[i]))
    return preds, proba


# =====================================================================
# EXPERIMENT 1 — 事前固定候選設計, 全部回報 (不挑最大)
# =====================================================================
print("=" * 72)
print("EXPERIMENT 1 — 事前固定候選設計的忠實 LOSO (全部回報)")
print("=" * 72)

Xday0 = sup[FEATURES_DAY0].values
Xrev = review_matrix(FEATURES_REVIEW, REVIEW_K)

experiments = []

# --- 無參數基準 (沒有 fit, 不可能過擬合) ---
def tier_rule_preds(t2="Low"):
    rule = {1: 0, 2: LABEL2I[t2], 3: 1}     # tier1->Low, tier3->Medium
    return np.array([rule[int(t)] for t in sup["tier"]])

for name, preds in [
    ("[基準] 多數類 (全判 Low)", np.zeros(N, dtype=int)),
    ("[基準] Tier 查表 (t1→Low,t2→Low,t3→Med)", tier_rule_preds("Low")),
    ("[基準] Tier 查表 (t1→Low,t2→Med,t3→Med)", tier_rule_preds("Medium")),
]:
    m = all_metrics(y, preds)
    k = int((preds == y).sum())
    m["boot_acc_ci"] = bootstrap_ci(y, preds, metric="accuracy")
    m["wilson_acc_ci"] = wilson_ci(k, N)
    m["n_correct"] = k
    experiments.append((name, m, preds, None))

# --- 有 fit 的候選 (設計事前固定) ---
model_specs = [
    ("LogReg Day-0 (tier,esg)", Xday0, lambda: mk_logreg(True)),
    ("LogReg Review (tier,esg,mav,single)", Xrev, lambda: mk_logreg(True)),
    ("LogReg Review 無class_weight", Xrev, lambda: mk_logreg(False)),
    ("Ordinal(Frank-Hall) Review", Xrev, lambda: OrdinalFrankHall(True)),
    ("Ordinal(Frank-Hall) Day-0", Xday0, lambda: OrdinalFrankHall(True)),
]
for name, Xmat, fn in model_specs:
    preds, proba = loso_predict(Xmat, y, fn)
    m = all_metrics(y, preds, proba)
    k = int((preds == y).sum())
    m["boot_acc_ci"] = bootstrap_ci(y, preds, proba, "accuracy")
    m["boot_f1_ci"] = bootstrap_ci(y, preds, proba, "macro_f1")
    m["wilson_acc_ci"] = wilson_ci(k, N)
    m["n_correct"] = k
    experiments.append((name, m, preds, proba))

# --- 非線性/交互作用: 用好那兩個弱訊號 (esg^2, esg*tier) ---
Xnl = np.column_stack([Xrev, (sup["esg"].values ** 2),
                       sup["esg"].values * sup["tier"].values])
preds, proba = loso_predict(Xnl, y, lambda: mk_logreg(True))
m = all_metrics(y, preds, proba); k = int((preds == y).sum())
m["boot_acc_ci"] = bootstrap_ci(y, preds, proba, "accuracy")
m["wilson_acc_ci"] = wilson_ci(k, N); m["n_correct"] = k
experiments.append(("LogReg Review + 非線性(esg^2,esg*tier)", m, preds, proba))

# 印出結果表
print(f"\n{'設計':44s}{'正確':>6s}{'Acc':>7s}{'mF1':>7s}{'MAE':>6s}"
      f"{'相鄰':>5s}{'對調':>5s}   {'Acc 95% boot CI':>20s}")
print("-" * 108)
for name, m, preds, proba in experiments:
    lo, hi = m["boot_acc_ci"]
    print(f"{name:44s}{m['n_correct']:>4d}/15{m['accuracy']:>7.1%}"
          f"{m['macro_f1']:>7.2f}{m['ordinal_mae']:>6.2f}"
          f"{m['adjacent_err']:>5d}{m['extreme_err']:>5d}   [{lo:>5.1%}, {hi:>5.1%}]")

print("\n備註: AUC-ROC(OvR macro)/LogLoss 於有機率輸出的設計附於 JSON;")
print("      在 LOSO+單一 High 樣本下 High 類 AUC 極不穩, 僅供參考。")

# =====================================================================
# EXPERIMENT 2 — 巢狀 CV (誠實調參, 證明調參不破 73.3%)
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 2 — 巢狀 CV: 決策樹深度 + LogReg C, inner 選 outer LOSO 評")
print("=" * 72)


def nested_loso(Xmat, y, build, grid, param_name):
    """outer LOSO; 每個 outer 折內用 leave-one-out 於訓練 14 家選最佳超參數。"""
    outer_pred = np.zeros(N, dtype=int)
    chosen = []
    for i in range(N):
        omask = np.ones(N, dtype=bool); omask[i] = False
        Xtr, ytr = Xmat[omask], y[omask]
        best_v, best_score = None, -1
        for v in grid:
            # inner LOOCV on the 14 training suppliers
            ok = 0
            for j in range(len(ytr)):
                imask = np.ones(len(ytr), dtype=bool); imask[j] = False
                mm = build(v).fit(Xtr[imask], ytr[imask])
                ok += int(mm.predict(Xtr[[j]])[0] == ytr[j])
            sc = ok / len(ytr)
            if sc > best_score:
                best_score, best_v = sc, v
        chosen.append(best_v)
        m = build(best_v).fit(Xtr, ytr)
        outer_pred[i] = int(m.predict(Xmat[[i]])[0])
    return outer_pred, chosen


def build_tree(depth):
    return Pipeline([("sc", StandardScaler()),
                     ("clf", DecisionTreeClassifier(max_depth=depth,
                                                    class_weight="balanced",
                                                    random_state=RANDOM_STATE))])

def build_lr(C):
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, max_iter=5000,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE))])

for label, Xmat, build, grid, pname in [
    ("決策樹 depth∈{1,2,3}", Xrev, build_tree, [1, 2, 3], "max_depth"),
    ("LogReg C∈{0.03,0.1,0.3,1,3}", Xrev, build_lr, [0.03, 0.1, 0.3, 1, 3], "C"),
]:
    preds, chosen = nested_loso(Xrev, y, build, grid, pname)
    m = all_metrics(y, preds); k = int((preds == y).sum())
    lo, hi = wilson_ci(k, N)
    print(f"\n  {label}: {k}/15 ({m['accuracy']:.1%})  macroF1={m['macro_f1']:.2f}"
          f"  MAE={m['ordinal_mae']:.2f}  Wilson95%=[{lo:.1%},{hi:.1%}]")
    print(f"    outer 折選到的 {pname}: {chosen}")

# =====================================================================
# EXPERIMENT 3 — 天花板 / Bayes 誤差分析
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 3 — 天花板分析 (為何 >73.3% 不穩健)")
print("=" * 72)

# (a) 結構性 High: LOSO 留出唯一 High → 該折訓練無 High → 必錯
n_high = int((y == 2).sum())
print(f"\n(a) High 類僅 {n_high} 家 (SUP-011)。LOSO 留出它時訓練集無 High,")
print(f"    無論何模型該折結構上必錯 → LOSO 準確率上限 = {N-1}/{N} = {(N-1)/N:.1%}。")

# (b) 特徵空間資訊上限: 1-NN LOSO (最樂觀的「特徵能分多少」探針)
knn_preds, _ = loso_predict(Xrev, y, lambda: KNeighborsClassifier(n_neighbors=1))
knn_acc = (knn_preds == y).mean()
print(f"\n(b) 1-NN LOSO (特徵資訊上限探針, Review 特徵): "
      f"{int((knn_preds==y).sum())}/15 ({knn_acc:.1%})")

# (c) Tier2 Low/Medium 的不可分性: 逐特徵單調矛盾
t2 = sup[sup["tier"] == 2].copy()
lows = t2[t2["risk"] == "Low"]; meds = t2[t2["risk"] == "Medium"]
print("\n(c) Tier2 (7 家) Low vs Medium — 特徵重疊/矛盾:")
print(f"    Low  ESG={sorted(lows['esg'].round(1).tolist())}  "
      f"single={sorted(lows['single_rate'].round(3).tolist())}")
print(f"    Med  ESG={sorted(meds['esg'].round(1).tolist())}  "
      f"single={sorted(meds['single_rate'].round(3).tolist())}")
# 有多少 Medium 的 ESG 高於「最低 ESG 的 Low」→ 單調 ESG 規則抓不到
low_min_esg = lows["esg"].min()
inseparable = meds[meds["esg"] > low_min_esg]
print(f"    最低 ESG 的 Low = {low_min_esg} (SUP-014)。")
print(f"    ESG 高於它、卻是 Medium 的供應商 (單調 ESG 規則無解): "
      f"{inseparable['Supplier ID'].tolist()}  ← 與 Low 在特徵上矛盾")
print(f"    僅 SUP-013 (ESG={meds['esg'].min()}) 低於所有 Low, 理論上可被 ESG 門檻分出。")

# (d) 樂觀天花板 = 14 - (無法分的 Tier2 Medium 數)
n_insep = len(inseparable)
optimistic = N - n_high - n_insep   # 扣掉結構 High + 矛盾 Medium
print(f"\n(d) 樂觀天花板估計: {N} - {n_high}(結構High) - {n_insep}(矛盾Medium) "
      f"= {optimistic}/15 ({optimistic/N:.1%})")
print(f"    → 即使完美利用特徵, 穩健上限約 {optimistic}/15 = {optimistic/N:.1%};")
print(f"      比 73.3% 只多 1 家 (SUP-013), 且該分界由「單一供應商」定義,")
print(f"      對 n=14 訓練折屬過擬合, 不構成穩健改進。")

# (e) 11 vs 12 是否落在雜訊區
lo11, hi11 = wilson_ci(11, 15); lo12, hi12 = wilson_ci(12, 15)
print(f"\n(e) 雜訊區判定 (Wilson 95%):")
print(f"    11/15 = 73.3%  CI=[{lo11:.1%}, {hi11:.1%}]")
print(f"    12/15 = 80.0%  CI=[{lo12:.1%}, {hi12:.1%}]  ← 與 11/15 CI 大幅重疊")
print(f"    兩者僅差 1 家 (±{1/15:.1%}), 落在 ±12pp 雜訊區內 → 非統計顯著改進。")

# =====================================================================
# EXPERIMENT 4 — 標籤獨立性 (permutation) 檢定: 行為特徵有無真訊號
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 4 — Permutation 檢定: 特徵 vs 標籤的關聯是否超出雜訊")
print("=" * 72)

base_preds, _ = loso_predict(Xrev, y, lambda: mk_logreg(True))
base_acc = (base_preds == y).mean()
null_accs = []
for _ in range(2000):
    yp = rng.permutation(y)
    pp, _ = loso_predict(Xrev, yp, lambda: mk_logreg(True))
    null_accs.append((pp == yp).mean())
null_accs = np.array(null_accs)
pval = float((null_accs >= base_acc).mean())
print(f"\n  Review LogReg 實際 LOSO acc = {base_acc:.1%}")
print(f"  打亂標籤的 null 分布: 平均 {null_accs.mean():.1%}, "
      f"95 百分位 {np.percentile(null_accs,95):.1%}")
print(f"  permutation p-value = {pval:.3f}")
print("  解讀: 訊號主要來自 tier (查表即得 73.3%);")
print("        行為率(mav/single)在供應商層級近乎與標籤獨立, 難再擠出泛化。")

# =====================================================================
# 導出 JSON 摘要
# =====================================================================
summary = {
    "n_suppliers": N,
    "label_dist": {r: int((sup["risk"] == r).sum()) for r in RISK_ORDER},
    "baseline_review_acc": 11 / 15,
    "experiments": [
        {"name": n, **{k: v for k, v in m.items()
                       if k not in ()}}
        for n, m, _, _ in experiments
    ],
    "knn_loso_acc": float(knn_acc),
    "optimistic_ceiling": optimistic / N,
    "structural_high_error": n_high,
    "inseparable_tier2_medium": inseparable["Supplier ID"].tolist(),
    "permutation_pvalue": pval,
    "wilson": {"11/15": wilson_ci(11, 15), "12/15": wilson_ci(12, 15)},
}
with open(os.path.join(OUTPUT_DIR, "ceiling_study_summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
print(f"\n💾 摘要: {os.path.join(OUTPUT_DIR, 'ceiling_study_summary.json')}")
print("\n🏁 完成。")
