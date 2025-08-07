#!/usr/bin/env python
# evaluate_best_models.py (RF for HOCM_vs_HNCM + 10 selected features + LR/SVM NaN policy)
# ────────────────────────────────────────────────────────────────────────────
#  Re-train the top model per task, report AUROC & AUPRC (95 % CI),
#  save OOF predictions, and draw overlay curves for all tasks.
#
#  Changes:
#   • For the HOCM_vs_HNCM task ONLY → use RandomForest **with best params**
#     and restrict X to the **10 selected features** you specified.
#   • For **Logistic Regression and SVM tasks** → drop the 3 columns with the
#     most NaNs, then drop rows with any remaining NaNs (matches prior policy).
#   • All other tasks remain as configured.
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# ─── 0) CONFIG ──────────────────────────────────────────────────────────────
DATA_PATH      = Path("full_cohort.csv")
RANDOM_STATE   = 42
N_BOOTSTRAPS   = 200
CV_OUTER       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

plt.rcParams.update({"figure.dpi": 160})
warnings.filterwarnings("ignore", category=FutureWarning)

# 10 selected features (YOUR column names)
SELECTED_10_FEATURES = [
    "R_S_ratio_I",
    "R_S_ratio_aVR",
    "mean_S_aVR",
    "mean_S_II",
    "mean_R_aVR",
    "qrs_end",
    "angle_e1_QRS_T_deg",
    "T_wave_dipolar_ln_mV",
    "norm_amp_e2_QRS",
    "ln_QRS_non_dipolar_norm",
    
]

# Optional: load RF best params from the previous tuning run (train_hocm_hncm_10feats)
# If present, we'll use its RF best params. Otherwise, fall back to a robust default.
RF_PARAMS_JSON_PATH = Path("results_hocm_hncm/best_params.json")
DEFAULT_RF_PARAMS_HOCM = {
    # Based on your CV results (common winners across folds)
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_split": 10,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

def _load_best_rf_params(json_path: Path) -> dict:
    if not json_path.exists():
        return DEFAULT_RF_PARAMS_HOCM
    try:
        with open(json_path, "r") as f:
            best = json.load(f)
        rf = best.get("RF", {})
        # In case keys are like "clf__max_depth" from a pipeline, strip the prefix
        cleaned = {}
        for k, v in rf.items():
            if "__" in k:
                k = k.split("__", 1)[1]
            cleaned[k] = v
        # Ensure essential defaults if missing
        for k, v in DEFAULT_RF_PARAMS_HOCM.items():
            cleaned.setdefault(k, v)
        return cleaned
    except Exception:
        return DEFAULT_RF_PARAMS_HOCM

BEST_RF_PARAMS_HOCM = _load_best_rf_params(RF_PARAMS_JSON_PATH)

# ─── 1) TASK DEFINITIONS ────────────────────────────────────────────────────
TASKS = {

    "HCM_vs_DCM_nonI": dict(                 # best AUROC 0.880 (LogReg)
        pos   = ["HOCM", "HNCM", "Probably HNCM", "HCM Unknown"],
        neg   = ["Dilated Non-Ischemic"],
        model = ("logreg", {
            "penalty": "l1", "C": 0.1, "class_weight": None,
            "solver": "liblinear", "max_iter": 2000,
        }),
    ),
    "HCM_vs_DCM_I": dict(                    # best AUROC 0.881 (LogReg)
        pos   = ["HOCM", "HNCM", "Probably HNCM", "HCM Unknown"],
        neg   = ["Dilated Ischemic"],
        model = ("logreg", {
            "penalty": "l1", "C": 0.1, "class_weight": None,
            "solver": "liblinear", "max_iter": 2000,
        }),
    ),

    # ── CHANGED: HOCM_vs_HNCM now uses RandomForest + ONLY the 10 selected features
    "HOCM_vs_HNCM": dict(
        pos   = ["HOCM"],
        neg   = ["HNCM"],
        model = ("rf", BEST_RF_PARAMS_HOCM),
    ),
}

# ─── 2) HELPERS ─────────────────────────────────────────────────────────────

def bootstrap_ci(metric_func, y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = N_BOOTSTRAPS, ci: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        scores.append(metric_func(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


def make_model(kind: str, params: dict):
    if kind == "xgb":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed in this environment")
        return XGBClassifier(
            objective="binary:logistic", eval_metric="auc", tree_method="hist",        # or "gpu_hist" in older versions
            device="cuda", n_jobs=-1, random_state=RANDOM_STATE, **params
        )
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", LogisticRegression(random_state=RANDOM_STATE, **params))
        ])
    if kind == "svm":
        return Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE))
        ])
    if kind == "rf":
        return Pipeline([
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced_subsample",
                **params,
            )),
        ])
    raise ValueError(f"Unknown model type {kind}")
def _prepare_train_oob_policy(train_df: pd.DataFrame,
                              oob_df: pd.DataFrame,
                              task_name: str,
                              model_kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same feature policies to TRAIN and OOB, using columns/thresholding
    chosen ONLY from TRAIN (to avoid leakage). Returns X_train, y_train, X_oob, y_oob."""
    # Copy to avoid side-effects
    tr = train_df.copy()
    ob = oob_df.copy()

    # Binary targets already present as 'target'
    y_tr = tr["target"].values
    y_ob = ob["target"].values

    # HOCM_vs_HNCM → ensure ratios; select fixed 10 features
    if task_name == "HOCM_vs_HNCM":
        tr = ensure_rs_ratios(tr)
        ob = ensure_rs_ratios(ob)
        missing_tr = [c for c in SELECTED_10_FEATURES if c not in tr.columns]
        missing_ob = [c for c in SELECTED_10_FEATURES if c not in ob.columns]
        if missing_tr or missing_ob:
            raise KeyError("Missing required columns for HOCM_vs_HNCM: "
                           + ", ".join(set(missing_tr + missing_ob)))
        X_tr = tr[SELECTED_10_FEATURES].values
        X_ob = ob[SELECTED_10_FEATURES].values
        return X_tr, y_tr, X_ob, y_ob

    # Other tasks:
    # For LR/SVM: drop top-3 NaN columns computed on TRAIN; then drop rows with any NaNs
    # For tree/XGB: use all remaining features as-is (consistent with your current policy)
    if model_kind in {"logreg", "svm"}:
        feats_tr = tr.drop(columns=["label", "target"], errors="ignore")
        na_counts = feats_tr.isna().sum().sort_values(ascending=False)
        to_drop = na_counts.head(3).index.tolist()
        if len(to_drop) > 0:
            tr = tr.drop(columns=to_drop)
            ob = ob.drop(columns=[c for c in to_drop if c in ob.columns], errors="ignore")

        # After column drops, drop rows with any NaNs *separately* in TRAIN and OOB
        tr = tr.dropna(axis=0, how="any")
        ob = ob.dropna(axis=0, how="any")

        # Targets may have changed length; re-extract
        y_tr = tr["target"].values
        y_ob = ob["target"].values
        X_tr = tr.drop(columns=["label", "target"], errors="ignore").values
        X_ob = ob.drop(columns=["label", "target"], errors="ignore").values
    else:
        # Tree/XGB tasks
        X_tr = tr.drop(columns=["label", "target"], errors="ignore").values
        X_ob = ob.drop(columns=["label", "target"], errors="ignore").values

    return X_tr, y_tr, X_ob, y_ob


def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Select threshold that maximizes Youden's J on the TRAIN set."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


def _sens_spec(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> tuple[float, float]:
    """Compute sensitivity and specificity at threshold thr."""
    y_hat = (y_prob >= thr).astype(int)
    tp = np.sum((y_true == 1) & (y_hat == 1))
    tn = np.sum((y_true == 0) & (y_hat == 0))
    fp = np.sum((y_true == 0) & (y_hat == 1))
    fn = np.sum((y_true == 1) & (y_hat == 0))
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return float(sens), float(spec)


def bootstrap_refit_oob_metrics(df_task: pd.DataFrame,
                                task_name: str,
                                model_kind: str,
                                params: dict,
                                n_boot: int = 1000,
                                ci: float = 0.95,
                                random_state: int = RANDOM_STATE) -> dict:
    """Bootstrap with refitting and OOB evaluation.
    Returns percentile CIs for AUC, AUPRC, Sensitivity, Specificity."""
    rng = np.random.default_rng(random_state)
    n = len(df_task)
    aucs, auprs, sens_list, spec_list = [], [], [], []

    # Pre-create a base model factory; each iteration builds a fresh estimator
    def _new_model():
        return make_model(model_kind, params)

    for _ in range(n_boot):
        # Sample indices with replacement for TRAIN
        train_idx = rng.choice(n, n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[train_idx] = False
        # If no OOB examples this round, skip (rare for moderate n)
        if not np.any(oob_mask):
            continue

        train_df = df_task.iloc[train_idx]
        oob_df   = df_task.iloc[oob_mask]

        # Apply task/policy-specific feature prep using TRAIN only
        try:
            X_tr, y_tr, X_ob, y_ob = _prepare_train_oob_policy(
                train_df, oob_df, task_name, model_kind
            )
        except Exception:
            # If a particular bootstrap split becomes invalid due to NA policies,
            # skip this iteration
            continue

        # Fit
        model = _new_model()
        try:
            model.fit(X_tr, y_tr)
        except Exception:
            continue

        # Predict
        try:
            y_prob_tr = model.predict_proba(X_tr)[:, 1]
            y_prob_ob = model.predict_proba(X_ob)[:, 1]
        except Exception:
            # Some models/pipelines may fail probabilistic prediction in edge cases
            continue

        # Train-chosen threshold (Youden)
        thr = _youden_threshold(y_tr, y_prob_tr)

        # OOB metrics
        if len(np.unique(y_ob)) < 2:
            # AUC undefined if OOB contains single class; skip
            continue

        aucs.append(roc_auc_score(y_ob, y_prob_ob))
        auprs.append(average_precision_score(y_ob, y_prob_ob))
        sens, spec = _sens_spec(y_ob, y_prob_ob, thr)
        if not np.isnan(sens):
            sens_list.append(sens)
        if not np.isnan(spec):
            spec_list.append(spec)

    def _ci(vals: list[float]) -> tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        a = (1 - ci) / 2
        return (float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a)))

    return {
        "AUC_CI":  _ci(aucs),
        "AUPRC_CI": _ci(auprs),
        "SENS_CI": _ci(sens_list),
        "SPEC_CI": _ci(spec_list),
        "n_effective": len(aucs)
    }


def ensure_rs_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure RS ratios exist for II and aVR; derive from mean_R_* / |mean_S_*| if missing."""
    out = df.copy()
    eps = 1e-6
    if "R_S_ratio_II" not in out.columns:
        if {"mean_R_II", "mean_S_II"}.issubset(out.columns):
            out["R_S_ratio_II"] = out["mean_R_II"] / (out["mean_S_II"].abs() + eps)
        elif {"RII", "SII"}.issubset(out.columns):
            out["R_S_ratio_II"] = out["RII"] / (out["SII"].abs() + eps)
        else:
            raise KeyError("Cannot derive R_S_ratio_II: need mean_R_II & mean_S_II (or RII/SII).")
    if "R_S_ratio_aVR" not in out.columns:
        if {"mean_R_aVR", "mean_S_aVR"}.issubset(out.columns):
            out["R_S_ratio_aVR"] = out["mean_R_aVR"] / (out["mean_S_aVR"].abs() + eps)
        elif {"RaVR", "SaVR"}.issubset(out.columns):
            out["R_S_ratio_aVR"] = out["RaVR"] / (out["SaVR"].abs() + eps)
        else:
            raise KeyError("Cannot derive R_S_ratio_aVR: need mean_R_aVR & mean_S_aVR (or RaVR/SaVR).")
    return out

# ─── 3) LOAD & GLOBAL PRE-FILTER ────────────────────────────────────────────
df_full = (
    pd.read_csv(DATA_PATH)
      .query("`cardiac complications` == False and `AI_certain` == True")
      .copy()
)

# Always-dropped columns (kept from your original script)
DROP_ALWAYS = [
    "surgery", "cardiac complications", "HCM_type", "path", "subject_id",
    "Atrial Fibrillation", "Atrial Flutter", "LBBB", "RBBB",
    "LAFB", "LPFB", "PVC",
]
df_full.drop(columns=[c for c in DROP_ALWAYS if c in df_full], inplace=True, errors="ignore")

# ─── 4) MAIN LOOP ────────────────────────────────────────────────────────────
records   = []        # for summary table
pred_rows = []        # for long CSV


for task_name, cfg in TASKS.items():
    pos, neg           = cfg["pos"], cfg["neg"]
    model_kind, params = cfg["model"]

    df_task = df_full[df_full["label"].isin(pos + neg)].copy()
    df_task["target"] = df_task["label"].isin(pos).astype(int)

    # Feature matrix per policy
    if task_name == "HOCM_vs_HNCM":
        # RF + only 10 selected features (derive RS ratios if needed)
        df_task = ensure_rs_ratios(df_task)
        missing = [c for c in SELECTED_10_FEATURES if c not in df_task.columns]
        if missing:
            raise KeyError("Missing required columns for HOCM_vs_HNCM: " + ", ".join(missing))
        X = df_task[SELECTED_10_FEATURES]
    else:
        # For LR/SVM tasks: drop top-3 NaN columns, then drop rows with any NaNs
        if model_kind in {"logreg", "svm"}:
            feats_only = df_task.drop(columns=["label", "target"], errors="ignore")
            na_counts = feats_only.isna().sum().sort_values(ascending=False)
            to_drop = na_counts.head(3).index.tolist()
            if len(to_drop) > 0:
                df_task = df_task.drop(columns=to_drop)
            df_task = df_task.dropna(axis=0, how="any")
            X = df_task.drop(columns=["label", "target"], errors="ignore")
        else:
            # For tree/XGB tasks, keep as-is (we'll impute or model handles NaNs)
            X = df_task.drop(columns=["label", "target"], errors="ignore")

    y = df_task["target"].values

    model = make_model(model_kind, params)

    # 5-fold OOF probabilities
    y_pred = cross_val_predict(
        model, X, y, cv=CV_OUTER, method="predict_proba", n_jobs=-1
    )[:, 1]

    # AUROC & AUPRC
    auc  = roc_auc_score(y, y_pred)
    aupr = average_precision_score(y, y_pred)

    boot = bootstrap_refit_oob_metrics(df_task, task_name, model_kind, params,
                                    n_boot= 1000, ci=0.95, random_state=RANDOM_STATE)
    auc_lo,  auc_hi  = boot["AUC_CI"]
    aupr_lo, aupr_hi = boot["AUPRC_CI"]
    sens_lo, sens_hi = boot["SENS_CI"]
    spec_lo, spec_hi = boot["SPEC_CI"]


    records.append(dict(
        Task=task_name,
        AUROC=f"{auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]",
        AUPRC=f"{aupr:.3f} [{aupr_lo:.3f}, {aupr_hi:.3f}]",
        Sens_Youden_OOB=f"[{sens_lo:.3f}, {sens_hi:.3f}]",
        Spec_Youden_OOB=f"[{spec_lo:.3f}, {spec_hi:.3f}]",
    ))  


    # Store per-row predictions
    pred_rows.extend(pd.DataFrame({
        "task":   task_name,
        "truth":  y,
        "prob":   y_pred,
    }).to_dict("records"))

# ─── 5) OUTPUTS ─────────────────────────────────────────────────────────────
# a) Summary table
summary_df = pd.DataFrame.from_records(records)
print("\n===== Cross-validated Performance =====")
print(summary_df.to_string(index=False))

# b) Save per-sample predictions
pred_df = pd.DataFrame.from_records(pred_rows)
pred_df.to_csv("predictions_all_tasks.csv", index=False)
print("\nPer-sample predictions → predictions_all_tasks.csv")

