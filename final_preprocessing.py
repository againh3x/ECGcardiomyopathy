import pandas as pd
import numpy as np

'''Final preprocessing for ML/Statistical analysis. ***Run ONLY AFTER all initial preprocessing and feature extraction is complete***'''
df = pd.read_csv('cohort.csv')

race_lower = df['race'].fillna('').str.lower()
df['race_black']    = race_lower.str.contains('black')
df['race_white']    = race_lower.str.contains('white')
df['race_hispanic'] = race_lower.str.contains('hispanic')
df['Male'] = df['gender'] == 'M'

df['admittime'] = pd.to_datetime(df['admittime']).dt.year
df['age'] = df['admittime'] - df['anchor_year'] + df['anchor_age']
df.drop(columns=['hadm_id','Unnamed: 0','admittime','dischtime','deathtime','admission_type','admit_provider_id','admission_location','discharge_location','insurance','language','marital_status','race','edregtime','edouttime','hospital_expire_flag', 'study_id', 'cart_id', 'ecg_time', 'report_0', 'report_1', 'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7', 'report_8', 'report_9', 'report_10', 'report_11', 'report_12', 'report_13', 'report_14', 'bandwidth', 'filtering', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod', 'Paced ECG', 'Faulty ECG'], inplace=True)

print("# ECG before 1 per patient:" ,len(df))
print("# Patients:" ,len(df['subject_id'].unique()))
df.replace(29999, np.nan, inplace=True)

def keep_least_nan_ecg(df):

    df['nan_count'] = df.isna().sum(axis=1)
    df_sorted = df.sort_values(by=['subject_id', 'nan_count'])
    df_unique = df_sorted.drop_duplicates(subset=['subject_id'], keep='first')
    df_unique = df_unique.drop(columns=['nan_count', 'subject_id'])
    
    return df_unique
df = keep_least_nan_ecg(df)
print("# ECG after 1 per patient:" ,len(df))
print("# Current Features", len(df.columns)-2)
df = df[['path'] + [col for col in df.columns if col != 'path']]


df['qrs_duration'] = df['qrs_end'] - df['qrs_onset']
df['p_duration'] = df['p_end'] - df['p_onset']

df.drop(columns=[
    'p_onset',
    'p_end',
    'qrs_onset',
    'qrs_end',
], inplace=True)


# 1. Define a function to identify P-wave related columns by name
def is_pwave_col(col: str) -> bool:
    c = col.lower()
    # Exclude obvious non-feature flags
    if c in {"sinus rhythm","atrial fibrillation","atrial flutter"}:
        return False
    return (
        "_p_" in c            # mean_P_I, dur_P_V6, etc.
        or c.startswith("dur_p_") 
        or c.startswith("dur_pr_")    # PR interval duration
        or c.startswith("dur_tpseg_") # TP segment
        or c.endswith("_pr")          # if you ever have a column like overall PR
        or c == "p_duration"
        or c == "p_axis"
    )

# 2. Build list of P-wave feature columns actually present in df
pwave_cols = [c for c in df.columns if is_pwave_col(c)]

print(f"Identified {len(pwave_cols)} P-wave related columns.")

# 3. Boolean mask of AF or AFlutter
af_mask = df.get("Atrial Fibrillation", False).astype(bool) | df.get("Atrial Flutter", False).astype(bool)

# 4. Set those columns to NaN where AF/AFlutter is True
df.loc[af_mask, pwave_cols] = np.nan

# (Optional) verify
print(f"Rows with AF/AFlutter: {af_mask.sum()}")
print("Example row after nulling P-wave cols:")
print(df.loc[af_mask, pwave_cols].head(1))
df.to_csv('final_df.csv', index=False)
