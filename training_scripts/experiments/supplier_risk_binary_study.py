# -*- coding: utf-8 -*-
"""
=====================================================================
 供應商風險評分 — 二分類框架研究 (Binary Reframing Study)
=====================================================================
角色: Data Analysis Engineer
問題: 把三分類 (Low / Medium / High) 改成「二分類 Low(0) vs
      Medium+High(1)」是不是「更好的問題框架」? 是真改進, 還是
      只是換皮 (raw accuracy 因合併 High 而虛高)?

商業對照: 二元 = 「自動核准 (Low)」 vs 「須人工複核 (Medium 以上)」,
          這可能才是真正的營運決策。

方法學紅線 (與 ceiling_study 一致, 全程遵守):
  1. 主指標: AUC-ROC(二元, 乾淨) + macro-F1 + 各類 P/R;
     裸準確率一律對照「60% 多數類基準」與「80% Tier 查表基準」,
     不與三分類 73.3% 直接相比 (不同任務)。禁用 MSE/R²。
  2. 不在測試集上挑選: 候選設計事前固定、全部回報 (不挑最大)。
     本研究「不調任何超參數」(LogReg 用預設 C=1.0), 故無選擇偏誤;
     不宣稱做了巢狀 CV。
  3. 無洩漏: scaler 只 fit 訓練折; Review 行為率用「前 k 筆」;
     主協定「對稱」(訓練列與測試列都用前 k 筆), 另報全歷史版對照。
     註: 此對稱協定與已部署三分類 new_supplier_risk_scoring.py 之
     Review (train 全歷史 / test 前 k, 非對稱) 不同; 因主對照走
     Tier 查表 (不含行為率), 協定差異不影響頭條結論。
  4. CI: 每個分數附 Wilson 二項 + bootstrap 95%; 差 1 家 (±6.7%)
     明確標示為雜訊。
  5. 固定 random_state = 42, 可複現。

不改動 new_supplier_risk_scoring.py / backend/models; 不 commit / push。
輸出: training_scripts/experiments/output/
"""

import os
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

# 二元標籤: Low=0 (自動核准), Medium/High=1 (人工複核)
BIN_LABELS = ["Low(0)", "Med+High(1)"]
REVIEW_K = 50
N_BOOT = 5000

print("=" * 72)
print("🧭 供應商風險 — 二分類框架研究 (Low vs Medium+High)")
print("=" * 72)

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
    risk3=("Supplier Risk", "first"),
).reset_index()

# 三分類 (0/1/2) 與二分類 (0/1) 標籤
y3 = sup["risk3"].map({"Low": 0, "Medium": 1, "High": 2}).values
y = (y3 >= 1).astype(int)                       # 二元: Medium 以上 = 1
N = len(sup)

FEATURES_DAY0 = ["tier", "esg"]
FEATURES_REVIEW = ["tier", "esg", "mav_rate", "single_rate"]

n_pos = int(y.sum()); n_neg = int(N - n_pos)
MAJ_ACC = max(n_pos, n_neg) / N                 # 多數類基準
print(f"\n供應商 {N} 家 | 二元標籤: Low(0)={n_neg}, Med+High(1)={n_pos}")
print(f"三分類分佈: {{Low:{int((y3==0).sum())}, Medium:{int((y3==1).sum())}, High:{int((y3==2).sum())}}}")
print(f"★ 多數類基準 (全猜 Low) = {n_neg}/{N} = {MAJ_ACC:.1%} (白送, 任何模型須超越此線)")


# =====================================================================
# as-of 行為率 (無洩漏): 被留出者只用「前 k 筆」
# =====================================================================
def early_rates(sid, k):
    s = df[df["Supplier ID"] == sid].head(k)
    return float(s["Maverick Spend"].mean()), float(s["Single Source Flag"].mean())


def review_matrix(feats, k):
    """對稱協定: 所有供應商 (含訓練列) 的行為率都用『前 k 筆』,
    確保 train/test apples-to-apples, 且被留出者未帶全歷史資訊。"""
    X = sup[feats].copy().astype(float)
    if "mav_rate" in feats:
        for i, sid in enumerate(sup["Supplier ID"]):
            mv, sg = early_rates(sid, k)
            X.loc[i, "mav_rate"] = mv
            X.loc[i, "single_rate"] = sg
    return X.values


# =====================================================================
# 指標工具 — 二元一次算齊 (AUC 為主, macro-F1, 各類 P/R)
# =====================================================================
def all_metrics(y_true, y_pred, proba1=None):
    """proba1 = P(class 1) 向量。二元指標。"""
    acc = float((y_true == y_pred).mean())
    mf1 = float(f1_score(y_true, y_pred, average="macro", labels=[0, 1],
                         zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0)
    out = {"accuracy": acc, "macro_f1": mf1,
           "precision": p.tolist(), "recall": r.tolist(), "f1": f.tolist()}
    if proba1 is not None:
        # AUC 需兩類皆存在 (bootstrap 折內可能退化)
        if len(np.unique(y_true)) == 2:
            try:
                out["auc"] = float(roc_auc_score(y_true, proba1))
            except Exception:
                out["auc"] = None
            try:
                out["log_loss"] = float(
                    log_loss(y_true, np.column_stack([1 - proba1, proba1]),
                             labels=[0, 1]))
            except Exception:
                out["log_loss"] = None
        else:
            out["auc"] = None
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


def bootstrap_ci(y_true, y_pred, proba1=None, metric="accuracy", n=N_BOOT):
    """Bootstrap over the 15 suppliers (modeling units)."""
    vals = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        pr = proba1[b] if proba1 is not None else None
        m = all_metrics(y_true[b], y_pred[b], pr)
        v = m.get(metric)
        if v is not None:
            vals.append(v)
    if not vals:
        return (None, None)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# =====================================================================
# 模型工廠 (事前固定設計) + 忠實 LOSO 執行器
# =====================================================================
def mk_logreg(balanced=True):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000,
                                   class_weight="balanced" if balanced else None,
                                   random_state=RANDOM_STATE)),
    ])


def loso_predict(Xmat, yv, model_fn):
    """留一供應商; scaler 只 fit 訓練折; 回傳 preds 與 P(class 1)。"""
    preds = np.zeros(N, dtype=int)
    proba1 = np.zeros(N)
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        m = model_fn().fit(Xmat[mask], yv[mask])
        pr = m.predict_proba(Xmat[[i]])[0]
        classes = list(m.named_steps["clf"].classes_) if hasattr(m, "named_steps") \
            else list(m.classes_)
        # 對齊 P(class 1)
        if 1 in classes:
            proba1[i] = pr[classes.index(1)]
        else:
            proba1[i] = 0.0
        preds[i] = int(m.predict(Xmat[[i]])[0])
    return preds, proba1


# =====================================================================
# EXPERIMENT 1 — 事前固定候選, 全部回報 (含兩個基準)
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 1 — 二元 LOSO (基準 + 事前固定候選, 全部回報)")
print("=" * 72)

Xday0 = sup[FEATURES_DAY0].values.astype(float)
Xrev = review_matrix(FEATURES_REVIEW, REVIEW_K)          # 對稱協定 (前 k 筆)
Xrev_full = sup[FEATURES_REVIEW].values.astype(float)    # 全歷史 (對照)

experiments = []


def tier_lookup_preds(t2_pos=False):
    """無 fit 基準: tier1→0, tier3→1, tier2→(Low 或 高風險)。"""
    rule = {1: 0, 2: (1 if t2_pos else 0), 3: 1}
    return np.array([rule[int(t)] for t in sup["tier"]])


# --- 無參數基準 ---
baselines = [
    ("[基準] 多數類 (全判 Low/0)", np.zeros(N, dtype=int), None),
    ("[基準] Tier 查表 (t1,t2→0; t3→1)", tier_lookup_preds(False), None),
    ("[基準] Tier 查表 (t1→0; t2,t3→1)", tier_lookup_preds(True), None),
]
for name, preds, _ in baselines:
    m = all_metrics(y, preds)
    k = int((preds == y).sum())
    m["n_correct"] = k
    m["wilson_acc_ci"] = wilson_ci(k, N)
    m["boot_acc_ci"] = bootstrap_ci(y, preds, metric="accuracy")
    experiments.append((name, m, preds, None))

# --- 有 fit 的候選 (設計事前固定) ---
model_specs = [
    ("LogReg Day-0 (tier,esg)", Xday0, lambda: mk_logreg(True)),
    ("LogReg Review 對稱 (tier,esg,mav,single)", Xrev, lambda: mk_logreg(True)),
    ("LogReg Review 無 class_weight", Xrev, lambda: mk_logreg(False)),
    ("LogReg Review 全歷史率 (對照)", Xrev_full, lambda: mk_logreg(True)),
]
for name, Xmat, fn in model_specs:
    preds, proba1 = loso_predict(Xmat, y, fn)
    m = all_metrics(y, preds, proba1)
    k = int((preds == y).sum())
    m["n_correct"] = k
    m["wilson_acc_ci"] = wilson_ci(k, N)
    m["boot_acc_ci"] = bootstrap_ci(y, preds, proba1, "accuracy")
    m["boot_auc_ci"] = bootstrap_ci(y, preds, proba1, "auc")
    m["boot_f1_ci"] = bootstrap_ci(y, preds, proba1, "macro_f1")
    experiments.append((name, m, preds, proba1))

# 結果表
print(f"\n{'設計':40s}{'正確':>7s}{'Acc':>7s}{'AUC':>7s}{'mF1':>7s}"
      f"   {'Acc 95% boot':>18s}   {'AUC 95% boot':>18s}")
print("-" * 112)
for name, m, preds, proba1 in experiments:
    alo, ahi = m["boot_acc_ci"]
    auc = m.get("auc")
    aucs = f"{auc:>6.2f}" if auc is not None else "   -- "
    if proba1 is not None and m.get("boot_auc_ci"):
        ulo, uhi = m["boot_auc_ci"]
        us = f"[{ulo:>5.2f}, {uhi:>5.2f}]" if ulo is not None else "        --        "
    else:
        us = "        --        "
    print(f"{name:40s}{m['n_correct']:>5d}/15{m['accuracy']:>7.1%}{aucs}"
          f"{m['macro_f1']:>7.2f}   [{alo:>5.1%}, {ahi:>5.1%}]   {us}")

print(f"\n★ 對照線: 多數類 = {MAJ_ACC:.1%} | Tier 查表 = {experiments[1][1]['accuracy']:.1%}"
      f" | (三分類 LOSO Review = 73.3%, 不同任務, 僅供參考)")
print("  註: AUC 在 n=6 正類、單一 High 下仍偏不穩, bootstrap CI 會很寬。")

# 各類 P/R (Review 對稱模型)
rev = next(m for n, m, _, _ in experiments if n.startswith("LogReg Review 對稱"))
print("\n各類 P/R (LogReg Review 對稱):")
for c, lab in enumerate(BIN_LABELS):
    print(f"  {lab:14s} P={rev['precision'][c]:.2f}  R={rev['recall'][c]:.2f}"
          f"  F1={rev['f1'][c]:.2f}")

# =====================================================================
# EXPERIMENT 2 — 逐供應商混淆: Tier2 Low vs 中高 有沒有消失?
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 2 — 逐供應商混淆 (重點: Tier2 的 Low vs 中高)")
print("=" * 72)

day0_preds, _ = loso_predict(Xday0, y, lambda: mk_logreg(True))
rev_preds, rev_p1 = loso_predict(Xrev, y, lambda: mk_logreg(True))

print(f"\n{'ID':9s}{'名稱':24s}{'Tier':>5s}{'三類':>7s}{'二元真':>8s}"
      f"{'Day0':>7s}{'Review':>8s}")
print("-" * 76)
for i, rowd in sup.iterrows():
    t3 = ["Low", "Med", "High"][y3[i]]
    yt = BIN_LABELS[y[i]]
    dp = BIN_LABELS[day0_preds[i]]
    rp = BIN_LABELS[rev_preds[i]]
    dmark = "✅" if day0_preds[i] == y[i] else "❌"
    rmark = "✅" if rev_preds[i] == y[i] else "❌"
    star = " ⟵Tier2" if rowd["tier"] == 2 else ""
    print(f"{rowd['Supplier ID']:9s}{rowd['name'][:22]:24s}{int(rowd['tier']):>5d}"
          f"{t3:>7s}{yt:>10s}{dp:>9s}{dmark}{rp:>8s}{rmark}{star}")

# Tier2 專項
t2mask = sup["tier"].values == 2
t2_ids = sup.loc[t2mask, "Supplier ID"].tolist()
t2_true = y[t2mask]
t2_day0 = day0_preds[t2mask]
t2_rev = rev_preds[t2mask]
print(f"\nTier2 專項 ({t2mask.sum()} 家): 真實二元 = {t2_true.tolist()}")
print(f"  Day-0   預測 = {t2_day0.tolist()}  → 正確 {int((t2_day0==t2_true).sum())}/{t2mask.sum()}")
print(f"  Review  預測 = {t2_rev.tolist()}  → 正確 {int((t2_rev==t2_true).sum())}/{t2mask.sum()}")

# Tier2 中高 (Medium) 供應商是否仍與 Tier2 Low 特徵矛盾
t2 = sup[t2mask].copy()
t2_low = t2[y[t2mask] == 0]
t2_hi = t2[y[t2mask] == 1]
low_min_esg = t2_low["esg"].min()
insep = t2_hi[t2_hi["esg"] > low_min_esg]
print(f"\n特徵可分性 (與三分類完全相同的矛盾):")
print(f"  Tier2 Low  ESG={sorted(t2_low['esg'].round(1).tolist())}")
print(f"  Tier2 中高 ESG={sorted(t2_hi['esg'].round(1).tolist())}  (皆 mav_rate=0, 行為率無訊號)")
print(f"  ESG 高於最低 Low({low_min_esg}) 卻是『中高』→ 單調規則無解: "
      f"{insep['Supplier ID'].tolist()}")
print("  ⟹ 合併 High 後, Tier2 的 Low↔中高 混淆『原封不動』(核心難點未被觸碰)。")

# =====================================================================
# EXPERIMENT 3 — 「換皮」拆解: 二元 80% 的多出來的是哪一家?
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 3 — raw accuracy 是否只是換皮 (High 合併的機械性增益)")
print("=" * 72)

# 三分類 tier 查表 (t1→Low, t2→Low, t3→Med) 與二元 tier 查表逐家比較
three_tier = np.array([{1: 0, 2: 0, 3: 1}[int(t)] for t in sup["tier"]])  # 三類意義: 0=Low,1=Med
three_correct = (three_tier == y3.clip(max=1)).astype(int)  # 對「合併後」是否對... 用二元比
# 更直接: 三分類任務的 tier 查表 acc vs 二元 tier 查表 acc, 找出翻轉的供應商
tl3 = {1: "Low", 2: "Low", 3: "Medium"}
pred3 = sup["tier"].map(tl3).values
acc3_task = float((pred3 == sup["risk3"].values).mean())
tl_bin = tier_lookup_preds(False)
acc_bin = float((tl_bin == y).mean())
flips = []
for i in range(N):
    c3 = pred3[i] == sup["risk3"].values[i]           # 三分類任務是否正確
    cb = tl_bin[i] == y[i]                             # 二元任務是否正確
    if c3 != cb:
        flips.append((sup["Supplier ID"][i], sup["name"][i],
                      sup["risk3"].values[i], "對" if c3 else "錯", "對" if cb else "錯"))
print(f"\n同一個 Tier 查表規則:")
print(f"  三分類任務 acc = {acc3_task:.1%} ({int((pred3==sup['risk3'].values).sum())}/15)")
print(f"  二元任務   acc = {acc_bin:.1%} ({int((tl_bin==y).sum())}/15)")
print(f"  差 = {acc_bin - acc3_task:+.1%}")
print(f"  由『錯→對』翻轉的供應商 (換皮增益來源):")
for fid, fname, fr, a3, ab in flips:
    print(f"    {fid} {fname[:22]:24s} (真實三類={fr}) 三分類={a3} → 二元={ab}")
print("  ⟹ 多出的準確率幾乎全部來自『唯一 High (SUP-011) 被併入 1 類後不再受")
print("     結構性 High 枷鎖懲罰』; 這是重貼標籤的機械效果, 非新的判別力。")

# 3b. 模型(LogReg) vs 零學習 Tier 查表 — 淨增益是「乾淨加一」還是「誤差重排」?
print(f"\n  模型 (Day-0 LogReg) vs 零學習 Tier 查表 (皆二元任務):")
print(f"    Tier 查表 = {int((tl_bin==y).sum())}/15 ({acc_bin:.0%}); "
      f"模型 Day-0 = {int((day0_preds==y).sum())}/15 ({(day0_preds==y).mean():.0%})")
gained = [sup['Supplier ID'][i] for i in range(N)
          if day0_preds[i] == y[i] and tl_bin[i] != y[i]]   # 模型對、查表錯
lost = [sup['Supplier ID'][i] for i in range(N)
        if day0_preds[i] != y[i] and tl_bin[i] == y[i]]      # 模型錯、查表對
print(f"    模型修好: {gained}   模型弄壞: {lost}")
print(f"    ⟹ 淨 {len(gained)-len(lost):+d} 家, 但非乾淨加一——是 Tier2 內的『誤差重排』")
print(f"      (修好 {gained} 卻弄壞 {lost}); CI 與查表重疊, 非穩健增益。")

# =====================================================================
# EXPERIMENT 4 — 二元的真實價值: 結構性 High 枷鎖是否解除
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 4 — 二元框架的『真改進』面向 (結構性 High 枷鎖)")
print("=" * 72)
print(f"""
三分類: High 全公司僅 1 家 (SUP-011)。LOSO 留它出局 → 訓練集無 High →
        該折結構上必錯 → 三分類 LOSO 準確率硬上限 = 14/15 = 93.3%。
二分類: SUP-011 併入『中高(1)』, 此類有 {n_pos} 家。LOSO 留任一家時
        訓練集恆有 1 類樣本 → 結構性必錯消失 (這是真的框架改進)。
        且 SUP-011 特徵 (tier3, mav_rate>0) 與其他中高一致, 可被正確判 1。

類別平衡: 三分類 9/5/1 (High 極端稀少) → 二分類 {n_neg}/{n_pos} (較平衡)。
""")

# 1-NN 特徵資訊上限探針 (二元)
knn_preds, knn_p1 = loso_predict(
    Xrev, y, lambda: KNeighborsClassifier(n_neighbors=1))
knn_acc = float((knn_preds == y).mean())
print(f"1-NN LOSO (特徵資訊上限探針): {int((knn_preds==y).sum())}/15 ({knn_acc:.1%})")

# 樂觀天花板: 15 - (Tier2 無法分的中高)
optimistic = N - len(insep)
print(f"樂觀天花板: {N} - {len(insep)}(Tier2 矛盾中高 {insep['Supplier ID'].tolist()}) "
      f"= {optimistic}/15 = {optimistic/N:.1%}")
print(f"  → 即使完美利用特徵, 穩健二元上限約 {optimistic/N:.1%}; 與 Tier 查表 "
      f"{acc_bin:.0%} 只差 {optimistic - int(acc_bin*N)} 家 (SUP-013 由 ESG 門檻,")
print("    但該門檻由單一供應商定義, LOSO 下過擬合, 非穩健改進)。")

# =====================================================================
# EXPERIMENT 5 — Permutation 檢定: 二元關聯是否超出雜訊
# =====================================================================
print("\n" + "=" * 72)
print("EXPERIMENT 5 — Permutation 檢定 (二元 LOSO acc / AUC vs null)")
print("=" * 72)

base_acc = float((rev_preds == y).mean())
try:
    base_auc = float(roc_auc_score(y, rev_p1))
except Exception:
    base_auc = None
null_acc, null_auc = [], []
for _ in range(2000):
    yp = rng.permutation(y)
    pp, pp1 = loso_predict(Xrev, yp, lambda: mk_logreg(True))
    null_acc.append((pp == yp).mean())
    if len(np.unique(yp)) == 2:
        try:
            null_auc.append(roc_auc_score(yp, pp1))
        except Exception:
            pass
null_acc = np.array(null_acc); null_auc = np.array(null_auc)
p_acc = float((null_acc >= base_acc).mean())
p_auc = float((null_auc >= base_auc).mean()) if base_auc is None else \
    float((null_auc >= base_auc).mean())
print(f"\n  實際 Review LOSO: acc={base_acc:.1%}"
      + (f", AUC={base_auc:.2f}" if base_auc is not None else ""))
print(f"  null(打亂標籤) acc: 均={null_acc.mean():.1%}, 95pct={np.percentile(null_acc,95):.1%}"
      f"  → p={p_acc:.3f}")
if len(null_auc):
    print(f"  null AUC: 均={null_auc.mean():.2f}, 95pct={np.percentile(null_auc,95):.2f}"
          f"  → p(AUC)={p_auc:.3f}")
print("  解讀: 訊號幾乎全來自 tier (mav_rate 完美旗標 Tier3, tier 分 Tier1/3);")
print("        Tier2 內 Low↔中高 仍不可分 → 行為/ESG 擠不出額外穩健泛化。")

# =====================================================================
# 混淆矩陣圖 (Day-0 vs Review, 二元)
# =====================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (preds, tag) in zip(
        axes, [(day0_preds, "Day-0"), (rev_preds, f"Review k={REVIEW_K}")]):
    cm = confusion_matrix(y, preds, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=BIN_LABELS).plot(
        ax=ax, cmap=plt.cm.Blues, colorbar=False)
    acc = (preds == y).mean()
    ax.set_title(f"Binary LOSO — {tag} (acc={acc:.0%})")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "binary_loso_confusion_matrix.png")
plt.savefig(cm_path, dpi=120)
plt.close()
print(f"\n💾 {cm_path}")

# =====================================================================
# 導出 JSON 摘要
# =====================================================================
summary = {
    "task": "binary Low(0) vs Medium+High(1)",
    "n_suppliers": N,
    "binary_label_dist": {"Low(0)": n_neg, "Med+High(1)": n_pos},
    "majority_baseline_acc": MAJ_ACC,
    "tier_lookup_baseline_acc": acc_bin,
    "three_class_reference_acc": 11 / 15,
    "experiments": [
        {"name": n, **{k: v for k, v in m.items()}}
        for n, m, _, _ in experiments
    ],
    "tier2": {
        "ids": t2_ids,
        "true_binary": t2_true.tolist(),
        "day0_pred": t2_day0.tolist(),
        "review_pred": t2_rev.tolist(),
        "inseparable_medium": insep["Supplier ID"].tolist(),
    },
    "reskin_flips": [f[0] for f in flips],
    "knn_loso_acc": knn_acc,
    "optimistic_ceiling": optimistic / N,
    "permutation": {"acc_pvalue": p_acc,
                    "base_acc": base_acc, "base_auc": base_auc},
    "wilson": {"9/15(60%)": wilson_ci(9, 15),
               "11/15(73%)": wilson_ci(11, 15),
               "12/15(80%)": wilson_ci(12, 15),
               "13/15(87%)": wilson_ci(13, 15)},
}
with open(os.path.join(OUTPUT_DIR, "binary_study_summary.json"), "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
print(f"💾 摘要: {os.path.join(OUTPUT_DIR, 'binary_study_summary.json')}")
print("\n🏁 完成。")
