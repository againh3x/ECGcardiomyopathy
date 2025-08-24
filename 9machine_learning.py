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
