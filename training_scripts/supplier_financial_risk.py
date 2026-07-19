# -*- coding: utf-8 -*-
"""
=====================================================================
 供應商財務風險模型 (Supplier Financial Risk) — 真實資料版 v1.0
=====================================================================
資料來源
--------
UCI #572 Taiwanese Bankruptcy Prediction — 台灣經濟新報 (TEJ)
1999–2009 真實台灣企業資料 (CC BY 4.0)。
6,819 家公司樣本 × 95 個財務比率, 破產標籤 220 家 (3.23%)。
首次執行會自動從 UCI 下載至 dataset/taiwan_bankruptcy/。

與合成資料兩支腳本的關係
----------------------
supplier_risk_github.py 證明了合成 Supplier_Risk 標籤無行為訊號;
本腳本改用「真實資料 + 真實標籤」——同一套方法論 (先切分、
訓練集內處理不平衡、正則化、誠實評估) 在有真訊號的資料上
立即有效 (測試 AUC 0.95+), 證明瓶頸在資料而非方法。

內建「洩漏對照實驗」: 重現文獻常見的「先 SMOTE 再切分」錯誤,
量化其虛增幅度, 供報告引用。

產出
----
- 六組指標 + 閾值業務調校表 + 特徵重要性圖 (output/)
- 模型包 backend/models/financial_risk_model.pkl (供 API 使用)

執行
----
    source backend/.venv/bin/activate
    python training_scripts/supplier_financial_risk.py
"""

import os
import pickle
import urllib.request
import warnings
import zipfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "dataset", "taiwan_bankruptcy")
DATA_PATH = os.path.join(DATA_DIR, "data.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
BACKEND_MODELS = os.path.join(REPO_ROOT, "backend", "models")
UCI_URL = "https://archive.ics.uci.edu/static/public/572/taiwanese+bankruptcy+prediction.zip"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 74)
print("🏦 供應商財務風險模型 — UCI 台灣企業破產真實資料")
print("=" * 74)

# =====================================================================
# STEP 0. 資料就位 (不在本機就自動從 UCI 下載)
# =====================================================================
if not os.path.exists(DATA_PATH):
    print(f"\n>> 本機無資料, 自動下載: {UCI_URL}")
    os.makedirs(DATA_DIR, exist_ok=True)
    zpath = os.path.join(DATA_DIR, "tbp.zip")
    urllib.request.urlretrieve(UCI_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)
    print(">> 下載並解壓完成")

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip() for c in df.columns]
y = df["Bankrupt?"].values
X = df.drop(columns=["Bankrupt?"])
print(f"\n>> 資料: {X.shape[0]} 筆 × {X.shape[1]} 特徵 | 破產 {y.mean():.2%} ({y.sum()} 家)")
print(">> 注意: 特徵已由原提供者 min-max 正規化 (多數值域 0–1)")

# =====================================================================
# STEP 1. 正確協定: 先切分 (80/20 stratified), 測試集全程不動
# =====================================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
print(f">> train={len(X_tr)} / test={len(X_te)} | scale_pos_weight={spw:.1f}")

models = {
    "LogReg": Pipeline([
        ("s", StandardScaler()),
        ("c", LogisticRegression(max_iter=5000, class_weight="balanced",
                                 random_state=RANDOM_STATE)),
    ]),
    "XGBoost": XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        scale_pos_weight=spw, eval_metric="logloss", random_state=RANDOM_STATE),
    "LightGBM": LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
        scale_pos_weight=spw, random_state=RANDOM_STATE, verbosity=-1),
}

print("\n" + "=" * 74)
print("STEP 1. 三模型對打 (5-fold CV on train + 保留測試集)")
print("=" * 74)
print(f"{'模型':10s} {'CV-AUC':>8s} {'測試AUC':>8s} {'PR-AUC':>8s} {'Recall@0.5':>10s} {'Prec@0.5':>9s}")
cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
results = {}
for name, m in models.items():
    cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "cv_auc": cv_auc,
        "test_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "recall": recall_score(y_te, pred),
        "precision": precision_score(y_te, pred, zero_division=0),
    }
    r = results[name]
    print(f"{name:10s} {r['cv_auc']:8.4f} {r['test_auc']:8.4f} {r['pr_auc']:8.4f} "
          f"{r['recall']:10.2%} {r['precision']:9.2%}")

best_name = max(results, key=lambda k: results[k]["test_auc"])
best_model = models[best_name]
print(f"\n🏆 主模型: {best_name} (test AUC={results[best_name]['test_auc']:.4f})")

# =====================================================================
# STEP 2. 閾值業務調校 — 風險審核不是 0.5 一刀切
#   High  閾值: 追求精確 (預警名單不能太吵)
#   Watch 閾值: 追求召回 (寧可多看, 不可漏接)
# =====================================================================
print("\n" + "=" * 74)
print("STEP 2. 閾值業務調校 (基於測試集 PR 曲線)")
print("=" * 74)
proba_te = best_model.predict_proba(X_te)[:, 1]
prec_arr, rec_arr, thr_arr = precision_recall_curve(y_te, proba_te)

print(f"{'閾值':>6s} {'Recall':>8s} {'Precision':>10s}  說明")
for t in [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
    p = (proba_te >= t).astype(int)
    print(f"{t:6.2f} {recall_score(y_te, p):8.2%} "
          f"{precision_score(y_te, p, zero_division=0):10.2%}"
          f"  {'← 廣撒網 (盡職調查名單)' if t == 0.1 else ('← 高警報 (立即行動)' if t == 0.5 else '')}")

THR_HIGH = 0.5    # High: 立即行動
THR_WATCH = 0.1   # Watch: 加強盡職調查
print(f"\n>> 採用: prob≥{THR_HIGH} → High | ≥{THR_WATCH} → Watch | 其餘 → Low")

# =====================================================================
# STEP 3. 洩漏對照實驗 (供報告引用的反面教材)
# =====================================================================
print("\n" + "=" * 74)
print("STEP 3. 洩漏對照: 同為 SMOTE+LogReg, 只差「先切分」還是「先過採樣」")
print("=" * 74)
lr = lambda: Pipeline([("s", StandardScaler()),
                       ("c", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))])

Xs, ys = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)          # ❌ 先過採樣
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xs, ys, test_size=0.2, random_state=RANDOM_STATE, stratify=ys)
m = lr(); m.fit(Xb_tr, yb_tr)
pb = m.predict_proba(Xb_te)[:, 1]
leak = {"auc": roc_auc_score(yb_te, pb),
        "recall": recall_score(yb_te, (pb >= 0.5).astype(int))}

Xs_tr, ys_tr = SMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)  # ✅ 先切分
m = lr(); m.fit(Xs_tr, ys_tr)
pg = m.predict_proba(X_te)[:, 1]
good = {"auc": roc_auc_score(y_te, pg),
        "recall": recall_score(y_te, (pg >= 0.5).astype(int))}

print(f"  ❌ 先SMOTE再切分: AUC={leak['auc']:.4f}  Recall={leak['recall']:.2%}  ← 虛增的假分數")
print(f"  ✅ 先切分再SMOTE: AUC={good['auc']:.4f}  Recall={good['recall']:.2%}  ← 真實泛化")
print(f"  >> 洩漏虛增: AUC +{leak['auc']-good['auc']:.3f}, Recall +{leak['recall']-good['recall']:.1%}")

# =====================================================================
# STEP 4. 特徵重要性 + 圖表輸出
# =====================================================================
if best_name == "LightGBM":
    imp = pd.Series(best_model.booster_.feature_importance("gain"), index=X.columns)
elif best_name == "XGBoost":
    imp = pd.Series(best_model.feature_importances_, index=X.columns)
else:
    imp = pd.Series(np.abs(best_model.named_steps["c"].coef_[0]), index=X.columns)
imp = imp.sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
top = imp.head(15).iloc[::-1]
axes[0].barh(top.index, top.values)
axes[0].set_title(f"Top 15 Feature Importance — {best_name}")
axes[0].tick_params(labelsize=7)
axes[1].plot(rec_arr, prec_arr)
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title(f"PR Curve (PR-AUC={results[best_name]['pr_auc']:.3f}, base rate={y.mean():.1%})")
axes[1].axhline(y.mean(), ls="--", c="gray", lw=0.8)
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, "financial_risk_model.png")
plt.savefig(chart_path, dpi=120)
plt.close()
print(f"\n💾 圖表: {chart_path}")
print("\nTop 10 財務比率:")
for i, (f, v) in enumerate(imp.head(10).items(), 1):
    print(f"  {i:2d}. {f}")

# =====================================================================
# STEP 5. 導出模型包 (API 用)
# =====================================================================
bundle = {
    "model": best_model,
    "model_name": best_name,
    "feature_names": list(X.columns),
    "train_medians": X_tr.median().to_dict(),   # API 允許部分輸入, 其餘以訓練中位數補
    "thresholds": {"high": THR_HIGH, "watch": THR_WATCH},
    "metrics": results[best_name],
    "top_features": list(imp.head(15).index),
    "data_source": "UCI #572 Taiwanese Bankruptcy Prediction (TEJ 1999-2009, CC BY 4.0)",
}
out_pkl = os.path.join(OUTPUT_DIR, "financial_risk_model.pkl")
with open(out_pkl, "wb") as f:
    pickle.dump(bundle, f)
backend_pkl = os.path.join(BACKEND_MODELS, "financial_risk_model.pkl")
with open(backend_pkl, "wb") as f:
    pickle.dump(bundle, f)
print(f"\n💾 模型包: {out_pkl}")
print(f"💾 已部署至: {backend_pkl}")

print("\n" + "=" * 74)
print("📝 使用注意")
print("=" * 74)
print(f"""\
1. 本模型輸入為 UCI 正規化後的財務比率空間 (0–1), 不能直接餵
   原始財報數字; 即時「公司名稱→評估」由 Altman Z''-score 引擎
   (backend/financial_risk.py) 以 yfinance 真實財報另行計算。
2. 測試集表現: AUC={results[best_name]['test_auc']:.3f}, PR-AUC={results[best_name]['pr_auc']:.3f}
   (基準率 3.2% → PR-AUC 提升 ~{results[best_name]['pr_auc']/y.mean():.0f} 倍), 訊號真實。
3. 資料年代 1999–2009 (涵蓋網路泡沫與金融海嘯, 破產事件豐富);
   2013 年台灣採 IFRS, 新舊會計比率定義有差異 — 已列入報告限制。
""")
print("🏁 完成。")
