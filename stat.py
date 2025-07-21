import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
'''One statistical analysis script for large feature analysis. This code does not convert T amplitudes to absolute value (paper figure utilized a different script)'''

FINAL_PATH   = "final_df.csv"
CONTROL_PATH = "control_group.csv"
AGE_COL_NAME = "age"     # Set to None if not present / not special
ID_COLS      = {"path", "Dilated", "Group"}  # columns to exclude from feature detection
SAVE_PATH    = "hcm_dcm_stats_with_control_reference.csv"

# ------------------ Load Data ----------------------
final_df   = pd.read_csv(FINAL_PATH)
control_df = pd.read_csv(CONTROL_PATH)

if "Dilated" not in final_df.columns:
    raise ValueError("Expected 'Dilated' column in final_df (1=DCM, 0=HCM).")

final_df["Group"] = np.where(final_df["Dilated"] == 1, "DCM", "HCM")

hcm = final_df[final_df["Group"] == "HCM"].copy()
dcm = final_df[final_df["Group"] == "DCM"].copy()

# ------------------ Feature Detection --------------
candidate_cols = [c for c in final_df.columns if c not in ID_COLS]

bool_cols = []
num_cols  = []

for c in candidate_cols:
    s = final_df[c]
    # Boolean-like: only 0/1/True/False (ignoring NaN) and <=2 unique non-NaN
    non_na = s.dropna()
    if non_na.empty:
        continue
    if non_na.isin([0,1,True,False]).all() and non_na.nunique() <= 2:
        bool_cols.append(c)
    elif pd.api.types.is_numeric_dtype(s):
        num_cols.append(c)

# Ensure age handled specially
if AGE_COL_NAME and AGE_COL_NAME in num_cols:
    num_cols.remove(AGE_COL_NAME)

# ------------------ Helper Functions ----------------
def rank_biserial_from_U(U, n1, n2):
    # U is the Mann–Whitney U for sample1 vs sample2
    return 1 - (2 * U) / (n1 * n2)

def hodges_lehmann(x, y):
    # median of all pairwise differences x_i - y_j
    return np.median(np.subtract.outer(x, y))

def fisher_odds_ratio(a_true, a_false, b_true, b_false):
    # Build table [[HCM_true, HCM_false],[DCM_true, DCM_false]]
    table = np.array([[a_true, a_false],
                      [b_true, b_false]], dtype=float)
    # continuity correction for zeros
    if (table == 0).any():
        table += 0.5
    return (table[0,0] * table[1,1]) / (table[0,1] * table[1,0])

def fmt_median_iqr(vals):
    if len(vals) == 0:
        return ""
    med = np.median(vals)
    q1  = np.percentile(vals,25)
    q3  = np.percentile(vals,75)
    return f"{med:.2f} [{q1:.2f}, {q3:.2f}]"

def compute_control_summary(feat):
    if feat not in control_df.columns:
        return ""
    v = control_df[feat].dropna().values
    if v.size == 0:
        return ""
    return fmt_median_iqr(v)

results = []

# ------------------ Numeric Features (Mann–Whitney) -------------
for feat in num_cols:
    x = hcm[feat].dropna().values
    y = dcm[feat].dropna().values
    if len(x) == 0 or len(y) == 0:
        continue

    try:
        U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        U, p = (np.nan, 1.0)

    hl = hodges_lehmann(x, y)                # HCM - DCM median difference (robust)
    rbc = rank_biserial_from_U(U, len(x), len(y)) if not np.isnan(U) else np.nan

    results.append({
        "Feature": feat,
        "HCM Cohort": fmt_median_iqr(x),
        "DCM Cohort": fmt_median_iqr(y),
        "Normal MIMIC ECG": compute_control_summary(feat),
        "Effect Size": f"HL diff={hl:.2f}; r={rbc:.3f}",
        "p_value": p
    })

# ------------------ Boolean Features (Fisher) -------------------
for feat in bool_cols:
    x_series = hcm[feat].dropna().astype(int)
    y_series = dcm[feat].dropna().astype(int)
    if len(x_series) == 0 or len(y_series) == 0:
        continue

    a_true = x_series.sum()
    a_false = len(x_series) - a_true
    b_true = y_series.sum()
    b_false = len(y_series) - b_true

    # Fisher exact test expects [[a_true, a_false],[b_true, b_false]]
    table = np.array([[a_true, a_false], [b_true, b_false]])
    try:
        OR, p = stats.fisher_exact(table, alternative="two-sided")
    except Exception:
        OR, p = (np.nan, 1.0)

    # Percent strings
    hcm_pct = f"{(a_true/len(x_series))*100:.1f}% ({a_true}/{len(x_series)})"
    dcm_pct = f"{(b_true/len(y_series))*100:.1f}% ({b_true}/{len(y_series)})"

    # Control prevalence if present
    if feat in control_df.columns:
        c_series = control_df[feat].dropna()
        if c_series.size:
            # interpret similarly (0/1)
            c_series = c_series.astype(int)
            c_true = c_series.sum()
            ctrl_pct = f"{(c_true/len(c_series))*100:.1f}% ({c_true}/{len(c_series)})"
        else:
            ctrl_pct = ""
    else:
        ctrl_pct = ""

    results.append({
        "Feature": feat,
        "HCM Cohort": hcm_pct,
        "DCM Cohort": dcm_pct,
        "Normal MIMIC ECG": ctrl_pct,
        "Effect Size": f"OR={OR:.2f}" if not np.isnan(OR) else "OR=NA",
        "p_value": p
    })

# ------------------ Age (special formatting) --------------------
if AGE_COL_NAME and AGE_COL_NAME in final_df.columns:
    x = hcm[AGE_COL_NAME].dropna().values
    y = dcm[AGE_COL_NAME].dropna().values
    if len(x) and len(y):
        # Welch t-test just for age difference (you can switch to MW if desired)
        tstat, p_age = stats.ttest_ind(x, y, equal_var=False)
        hl_age = hodges_lehmann(x, y)
        hcm_age = f"{x.mean():.1f} ± {x.std(ddof=1):.1f}"
        dcm_age = f"{y.mean():.1f} ± {y.std(ddof=1):.1f}"
        results.append({
            "Feature": AGE_COL_NAME,
            "HCM Cohort": hcm_age,
            "DCM Cohort": dcm_age,
            "Normal MIMIC ECG": "",  # no control age
            "Effect Size": f"HL diff={hl_age:.2f}",
            "p_value": p_age
        })

# ------------------ Multiple Testing Correction -----------------
# Collect p-values & apply FDR (includes age & booleans)
pvals = [r["p_value"] for r in results]
reject, qvals, *_ = multipletests(pvals, method="fdr_bh")
for r, q, rej in zip(results, qvals, reject):
    r["q_FDR"] = q
    r["Significant (q<0.05)"] = "Yes" if q < 0.05 else "No"

# ------------------ Build / Sort Table --------------------------
res_df = pd.DataFrame(results)
res_df = res_df.sort_values("q_FDR", kind="mergesort")  # stable

# Optional: nicer column order
col_order = ["Feature",
             "HCM Cohort","DCM Cohort","Normal MIMIC ECG",
             "Effect Size","p_value","q_FDR","Significant (q<0.05)"]
res_df = res_df[col_order]

# Save
res_df.to_csv(SAVE_PATH, index=False)
print(f"Saved stats table to: {SAVE_PATH}")
print(res_df.head(15))
