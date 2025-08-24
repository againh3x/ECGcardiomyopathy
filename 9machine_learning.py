#!/usr/bin/env python
# evaluate_best_models.py ────────────────────────────────────────────────────
#   • Evaluates both L1-penalised Logistic-Regression and XGBoost.
#   • HOCM vs HNCM uses the 10 hand-picked features; all rows with NaNs are
#     dropped.  Other tasks drop the 3 columns with most NaNs, then drop any
#     remaining NaN rows (matches prior policy).
#   • Outputs: cross-validated AUROC/AUPRC and per-sample OOF probs.
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.linear_model   import LogisticRegression
from sklearn.metrics        import (
    roc_auc_score, average_precision_score, roc_curve
)

from xgboost import XGBClassifier

# ─── CONFIG ────────────────────────────────────────────────────────────────
DATA_PATH     = Path("full_cohort1.csv")
RANDOM_STATE  = 1
CV_OUTER      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

warnings.filterwarnings("ignore", category=FutureWarning)

SELECTED_10_FEATURES = [
    "R_S_ratio_II", "R_S_ratio_aVR",  "mean_S_II", "mean_S_aVR", "mean_R_aVR", "mean_R_V6", "mean_R_V2", "RS_skew_aVR", "ST_slope_aVR",
    "qrs_end", "angle_e1_QRS_T_deg", "spatial_mean_P_QRS_angle_deg", "R_wave_amp_lead_Y_mV",
    "norm_amp_e2_QRS", "ln_QRS_non_dipolar_norm", 
]

TASKS = {
    "HCM_vs_DCM_nonI": dict(
        pos   = ["HOCM", "HNCM","HCM unknown"],
        neg   = ["DCM-NI"],
    ),
    "HCM_vs_DCM_I": dict(
        pos   = ["HOCM", "HNCM", "HCM unknown"],
        neg   = ["DCM-I"],
    ),
    "HOCM_vs_HNCM": dict(
        pos   = ["HOCM"],
        neg   = ["HNCM"],
    ),
}

# ─── HELPERS ───────────────────────────────────────────────────────────────
def make_logreg():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(
            penalty="l1", C=0.1, solver="liblinear",
            class_weight="balanced", max_iter=3000,
            random_state=RANDOM_STATE)
        ),
    ])

def make_xgb():
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1
    )

def youden_thr(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return thr[np.argmax(tpr - fpr)]

def sens_spec(y, p, thr):
    yhat = (p >= thr)
    tp = np.sum((y==1) &  yhat)
    tn = np.sum((y==0) & ~yhat)
    fp = np.sum((y==0) &  yhat)
    fn = np.sum((y==1) & ~yhat)
    sens = tp / (tp+fn) if tp+fn else np.nan
    spec = tn / (tn+fp) if tn+fp else np.nan
    return sens, spec

def ensure_rs(df):
    out = df.copy()
    eps = 1e-6
    if "R_S_ratio_aVR" not in out and {"mean_R_aVR","mean_S_aVR"}.issubset(out):
        out["R_S_ratio_aVR"] = out["mean_R_aVR"] / (out["mean_S_aVR"].abs()+eps)
    if "R_S_ratio_I" not in out and {"mean_R_I","mean_S_I"}.issubset(out):
        out["R_S_ratio_I"] = out["mean_R_I"] / (out["mean_S_I"].abs()+eps)
    return out

# ─── LOAD DATA ─────────────────────────────────────────────────────────────
df_full = pd.read_csv(DATA_PATH)
df_full.drop(columns=["subject_id","hadm_id", "path"], errors="ignore", inplace=True)

# ─── MAIN ──────────────────────────────────────────────────────────────────
records, pred_rows = [], []

for name, cfg in TASKS.items():
    df = df_full[df_full["Final_Label"].isin(cfg["pos"] + cfg["neg"])].copy()
    df["target"] = df["Final_Label"].isin(cfg["pos"]).astype(int)

    if name == "HOCM_vs_HNCM":                       # fixed 10-feature subset
        df = ensure_rs(df).dropna()
        X = df[SELECTED_10_FEATURES].values
    else:                                            # drop-NaN policy
        feats = df.drop(columns=["Final_Label","target"], errors="ignore")
        drop_cols = feats.isna().sum().sort_values(ascending=False).head(3).index
        df = df.drop(columns=drop_cols).dropna()
        X = df.drop(columns=["Final_Label","target"], errors="ignore").values

    y = df["target"].values

    for model_name, model in [("LogReg", make_logreg()), ("XGBoost", make_xgb())]:
        y_pred = cross_val_predict(model, X, y, cv=CV_OUTER,
                                   method="predict_proba", n_jobs=-1)[:,1]

        auc  = roc_auc_score(y, y_pred)
        aupr = average_precision_score(y, y_pred)
        
        thr = youden_thr(y, y_pred)
     
       
        sens, spec = sens_spec(y, y_pred, thr)

        records.append(dict(
            Task=f"{name}_{model_name}",
            AUROC=f"{auc:.3f}",
            AUPRC=f"{aupr:.3f}",
            Sensitivity=f"{sens:.3f}",
            Specificity=f"{spec:.3f}",
        ))

        pred_rows.extend(pd.DataFrame({
            "task": f"{name}_{model_name}", "truth": y, "prob": y_pred
        }).to_dict("records"))

# ─── OUTPUTS ───────────────────────────────────────────────────────────────
summary = pd.DataFrame(records)
print("\n===== Cross-validated Performance =====")
print(summary.to_string(index=False))

pd.DataFrame(pred_rows).to_csv("predictions_all_tasks.csv", index=False)
print("\nPer-sample predictions → predictions_all_tasks.csv")



# ─────────────────────────────────────────────────────────────────────────────
#   ROC (by task) with both models per panel + PR figure.
# - Uses pooled/OOF predictions (no retraining).
# - Legend shows model + AUC rounded to ROUND_TO.
# ─────────────────────────────────────────────────────────────────────────────
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

# ================= USER SETTINGS =================
ROUND_TO = 2
FIG_W, FIG_H = 12, 4   # 3 panels side-by-side
LEGEND_TITLE = "Model"
TASK_LABELS = {  # pretty names for panels
    "HCM_vs_DCM_nonI": "HCM vs DCM-NI",
    "HCM_vs_DCM_I":    "HCM vs DCM-I",
    "HOCM_vs_HNCM":    "HOCM vs HNCM",
}
MODEL_LABELS = {
    "LogReg": "Logistic Regression",
    "LR":     "Logistic Regression",
    "Logistic": "Logistic Regression",
    "XGBoost": "XGBoost",
    "XGB":     "XGBoost",
}
MODEL_ORDER = ["Logistic Regression", "XGBoost"]  # controls legend order
TASK_ORDER  = ["HCM_vs_DCM_nonI", "HCM_vs_DCM_I", "HOCM_vs_HNCM"]  # panel order
SAVE_ROC = "roc_by_task.png"
SAVE_PR  = "pr_by_task.png"   # supplement

# ================= DATA LOADING ===================
def _load_predictions():
    if "pred_rows" in globals() and isinstance(pred_rows, (list, tuple)) and len(pred_rows) > 0:
        df_ = pd.DataFrame(pred_rows)
    elif os.path.exists("predictions_all_tasks.csv"):
        df_ = pd.read_csv("predictions_all_tasks.csv")
    else:
        raise FileNotFoundError("Need `pred_rows` in memory or 'predictions_all_tasks.csv' on disk.")
    need = {"task","truth","prob"}
    if not need.issubset(df_.columns):
        raise ValueError(f"Predictions missing required columns: {need - set(df_.columns)}")
    return df_

df = _load_predictions().copy()

# If there's no 'model' column, try to infer from task suffixes.
if "model" not in df.columns:
    def parse_task_model(t):
        # Accept common suffixes: _LogReg, _LR, _Logistic, _XGBoost, _XGB, etc.
        m = re.search(r"_(LogReg|LR|Logistic|XGBoost|XGB)$", t)
        if m:
            raw = m.group(1)
            model = MODEL_LABELS.get(raw, raw)
            base = t[: -len(m.group(0))]  # strip suffix
        else:
            # Fallback: try to guess; everything goes to Logistic Regression
            model = "Logistic Regression"
            base = t
        return base, model

    parsed = df["task"].map(parse_task_model)
    df["task_base"] = parsed.map(lambda x: x[0])
    df["model"]     = parsed.map(lambda x: x[1])
else:
    df["task_base"] = df["task"]

# Keep only tasks we care about (if present)
df = df[df["task_base"].isin(TASK_ORDER)].copy()

# compute ROC 
def _roc(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    return fpr, tpr, auc

def _pr(y, p):
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)
    return rec, prec, ap

plt.rcParams.update({
    "figure.dpi": 160,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
})

fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
if len(TASK_ORDER) == 1:
    axes = [axes]

summary_rows = []

for ax, task_key in zip(axes, TASK_ORDER):
    d_task = df[df["task_base"] == task_key]
    # ensure deterministic legend order
    for model_name in MODEL_ORDER:
        d = d_task[d_task["model"] == model_name]
        if d.empty:
            continue
        y = d["truth"].to_numpy()
        p = d["prob"].to_numpy()
        fpr, tpr, auc = _roc(y, p)
        ax.plot(fpr, tpr, linewidth=2, label=f"{model_name}: AUC {auc:.{ROUND_TO}f}")
        summary_rows.append({
            "Task": TASK_LABELS.get(task_key, task_key),
            "Model": model_name,
            "AUROC": round(float(auc), ROUND_TO),
        })

    ax.plot([0,1],[0,1], linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_xlabel("1 – Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(TASK_LABELS.get(task_key, task_key))
    leg = ax.legend(title=LEGEND_TITLE, loc="lower right", frameon=True, fancybox=True, ncol=1)
    if leg and leg.get_title(): leg.get_title().set_fontsize(10)

fig.tight_layout()
fig.savefig(SAVE_ROC, dpi=300)
print(f"Saved ROC figure: {SAVE_ROC}")

# optional PR plots
fig2, axes2 = plt.subplots(1, len(TASK_ORDER), figsize=(FIG_W, FIG_H), sharex=True, sharey=True)
if len(TASK_ORDER) == 1:
    axes2 = [axes2]

for ax, task_key in zip(axes2, TASK_ORDER):
    d_task = df[df["task_base"] == task_key]
    for model_name in MODEL_ORDER:
        d = d_task[d_task["model"] == model_name]
        if d.empty:
            continue
        y = d["truth"].to_numpy()
        p = d["prob"].to_numpy()
        rec, prec, ap = _pr(y, p)
        ax.plot(rec, prec, linewidth=2, label=f"{model_name}: AP {ap:.{ROUND_TO}f}")
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(TASK_LABELS.get(task_key, task_key))
    leg = ax.legend(title=LEGEND_TITLE, loc="lower left", frameon=True, fancybox=True, ncol=1)
    if leg and leg.get_title(): leg.get_title().set_fontsize(10)

fig2.tight_layout()
fig2.savefig(SAVE_PR, dpi=300)
print(f"Saved PR figure: {SAVE_PR}")

#summary table
perf_df = pd.DataFrame(summary_rows)
perf_df = perf_df.pivot(index="Task", columns="Model", values="AUROC")
print(perf_df.round(2))

