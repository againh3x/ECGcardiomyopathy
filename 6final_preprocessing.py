import pandas as pd
import numpy as np
df = pd.read_csv('full_cohort.csv')
rec_df = pd.read_csv('data/record_list.csv')
pd.set_option('display.max_colwidth', None)  # Show full text in DataFrame
pd.set_option('display.max_rows', None)  # Show all rows in DataFrame
pd.set_option('display.max_columns', None)  # Show all columns in DataFrame

df['label'] = ''
df['AI_certain'] = True
df['HCM_type'] = ''
df['cardiac complications'] = False

mask = (df['phenotype'] == 'ischemic')
df.loc[mask, 'label'] = 'Dilated Ischemic'

mask = (df['phenotype'] == 'unknown') & (df['Ischemic'] == True) & (df['cardiomyopathy'] == 'Dilated')
df.loc[mask, 'label'] = 'Dilated Ischemic'
df.loc[mask, 'AI_certain'] = False

mask = (df['phenotype'] == 'non-ischemic') & (df['Ischemic'] == False)
df.loc[mask, 'label'] = 'Dilated Non-Ischemic'

mask = (df['phenotype'] == 'non-ischemic') & (df['Ischemic'] == True)
df.loc[mask, 'label'] = 'Dilated Unknown'

mask = (df['phenotype'] == 'unknown') & (df['Ischemic'] == False) & (df['cardiomyopathy'] == 'Dilated')
df.loc[mask, 'label'] = 'Dilated Non-Ischemic'
df.loc[mask, 'AI_certain'] = False


mask = (df['obstructive'] == 'Obstructive')
df.loc[mask, 'label'] = 'HOCM'

mask = (df['obstructive'] == 'Non-obstructive')
df.loc[mask, 'label'] = 'HNCM'

mask = (df['obstructive'] == 'Probably non-obstructive')
df.loc[mask, 'label'] = 'Probably HNCM'

mask = (df['obstructive'] == 'Unknown')
df.loc[mask, 'label'] = 'HCM Unknown'

mask = (df['phenotype'] == 'apical')
df.loc[mask, 'HCM_type'] = 'Apical'

mask = (df['phenotype'] == 'septal')
df.loc[mask, 'HCM_type'] = 'Septal'

mask = (df['Dilated'] == False) & (df['cardiomyopathy'] == 'Neither')
df.loc[mask, 'label'] = 'HCM Unknown'
df.loc[mask, 'AI_certain'] = False

mask = (df['Dilated'] == True) & (df['cardiomyopathy'] == 'Neither')
df.loc[mask, 'label'] = 'Dilated Unknown'
df.loc[mask, 'AI_certain'] = False



mask = (df['LBBB'] == True) | (df['RBBB'] == True) | (df['IVCD'] == True) | (df['LAFB'] == True) | (df['LPFB'] == True) | (df['Atrial Fibrillation'] == True) | (df['Atrial Flutter'] == True) | (df['PVC'] == True) 
df.loc[mask, 'cardiac complications'] = True
print(len(df))
df.drop(columns='ecg_time', inplace=True, errors='ignore')
df = pd.merge(df, rec_df[['path', 'ecg_time']], on='path', how='left')
print(len(df))
surg = df[df['surgery']]
earliest_surg_ecg = (
    surg.sort_values(['hadm_id', 'ecg_time'])      # chronological
         .groupby('hadm_id', as_index=False)        # one per admission
         .head(1)                                   # earliest only
)
nonsurg = df[~df['surgery']]
df = pd.concat([earliest_surg_ecg, nonsurg], ignore_index=True)
print(len(df))

print(len(df[df['label'] == 'HOCM']['subject_id'].unique()))
print(len(df[df['label'] == 'HNCM']['subject_id'].unique()))
print(len(df[df['label'] == 'Dilated Non-Ischemic']['subject_id'].unique()))
print(len(df[df['label'] == 'Dilated Ischemic']['subject_id'].unique()))
print(len(df[df['label'] == 'Dilated Unknown']['subject_id'].unique()))
print(len(df[df['label'] == 'HCM Unknown']['subject_id'].unique()))
print(len(df[df['label'] == 'Probably HNCM']['subject_id'].unique()))

label_rank = {
    "HOCM": 0, "HNCM": 0,
    "Dilated Non-Ischemic": 1,
    "Dilated Ischemic": 2,
    "Dilated Unknown": 3,
    "HCM Unknown": 4, "Probably HNCM": 4      # any leftovers rank worst
}
df["label_rank"] = df["label"].map(label_rank).fillna(5).astype(int)

# 2‑b  True = preferred → convert so 0 = best, 1 = worse
df["cardiac_complications_rank"] = df["cardiac complications"].astype(int)     # False→0  True→1
df["surgery_rank"]                = df["surgery"].astype(int)                  # False→0  True→1
df["AI_rank"]                     = (~df["AI_certain"]).astype(int)            # True→0  False→1

# 2‑c  # of NaNs in numeric cols (lower is better)
num_cols            = df.select_dtypes(include=[np.number]).columns
df["nan_count"]     = df[num_cols].isna().sum(axis=1)

# 2‑d  QRS duration (already numeric)
df["qrs_duration"]  = pd.to_numeric(df["qrs_duration"], errors="coerce")

# ---------------------------
# 3.  Sort by all priority keys
# ---------------------------
df_sorted = (
    df.sort_values(
        by=[
            "cardiac_complications_rank",  # 0 before 1
            "surgery_rank",                # 0 before 1
            "label_rank",                  # smaller rank wins
            "AI_rank",                     # 0 before 1
            "nan_count",                   # fewer NaNs first
            "qrs_duration",                # shorter QRS first
            "ecg_time"                     # earliest if still tied
        ],
        ascending=True                     # all keys ascending
    )
)

# ---------------------------
# 4.  Keep the first ECG per patient
# ---------------------------
df = (
    df_sorted
      .groupby("subject_id", as_index=False)
      .first()                              # first row after priority sort
)

print(len(df[df['label'] == 'HOCM']['subject_id'].unique()))
print(len(df[df['label'] == 'HNCM']['subject_id'].unique()))
print(len(df[df['label'] == 'Dilated Non-Ischemic']['subject_id'].unique()))
print(len(df[df['label'] == 'Dilated Ischemic']['subject_id'].unique()))
print(len(df[df['label'] == 'Dilated Unknown']['subject_id'].unique()))
print(len(df[df['label'] == 'HCM Unknown']['subject_id'].unique()))
print(len(df[df['label'] == 'Probably HNCM']['subject_id'].unique()))

race_lower = df['race'].fillna('').str.lower()
df['race_black']    = race_lower.str.contains('black')
df['race_white']    = race_lower.str.contains('white')
df['race_hispanic'] = race_lower.str.contains('hispanic')
df['Male'] = df['gender'] == 'M'

df['admittime'] = pd.to_datetime(df['admittime']).dt.year
df['age'] = df['admittime'] - df['anchor_year'] + df['anchor_age']

df.replace(29999, np.nan, inplace=True)
df['p_duration'] = df['p_end'] - df['p_onset']

#VCG P-wave features are already set to NaN if afib/flutter present during feature extraction
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


df.drop(columns=['Unnamed: 0','admittime','dischtime','deathtime','admission_type','admit_provider_id','admission_location','discharge_location','insurance','language','marital_status','race','edregtime','edouttime','hospital_expire_flag', 'study_id', 'cart_id', 'ecg_time', 'report_0', 'report_1', 'report_2', 'report_3', 'report_4', 'report_5', 'report_6', 'report_7', 'report_8', 'report_9', 'report_10', 'report_11', 'report_12', 'report_13', 'report_14', 'bandwidth', 'filtering', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod', 'Paced ECG', 'Faulty ECG', 'cardiomyopathy', 'Dilated', 'Ischemic', 'phenotype', 'obstructive', 'ecg_time','label_rank','cardiac_complications_rank','surgery_rank','AI_rank','nan_count', 'p_onset'], inplace=True, errors='ignore')
df = df[['path'] + [col for col in df.columns if col != 'path']]
print(len(df.columns))

path = ['path']
subject = ['subject_id', 'hadm_id']

# 2. Demographic features
demographics = ['surgery', 'age', 'Male', 'race_black', 'race_white', 'race_hispanic']

# 3. Basic rhythm, timing, and axis measurements


# 4. Clinical/ECG condition flags
condition_flags = [
    'Abnormal ECG','Borderline ECG','Sinus Rhythm','Sinus bradycardia',
    'Sinus Tachycardia','Low Precordial Voltage','Low Limb Voltage',
    'LVH','RVH','LAE','RAE','ST Elevation','ST Depression','ST Changes',
    'T Wave Changes','Atrial Fibrillation','Atrial Flutter','LBBB','RBBB',
    'LAFB','LPFB','LAD','RAD','Right Atrial Abnormality',
    'Left Atrial Abnormality','PVC','RVR','SVR'
]

base_features = [
    'rr_interval','p_end','qrs_onset','qrs_end','t_end','p_duration',
    'p_axis','qrs_axis','t_axis'
]

# 5. ECG lead‑based features (grouped by lead: I, II, III, aVR, aVL, aVF, V1–V6)
leads = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
lead_metrics = [
    'mean_Q','mean_R','mean_S','mean_T','mean_P',
    'R_S_ratio','T_R_ratio','QRS_area','ST_slope','RS_skew',
    'mean_freq','median_freq','skewness_time','kurtosis_time'
]
lead_columns = [f'{metric}_{lead}' for lead in leads for metric in lead_metrics]

# 6. Duration and interval features
duration_features = [
    'T_index','dur_QSpk','dur_R','dur_STpk','dur_spk-ton',
    'dur_T','dur_TPseg','dur_P','dur_QT','dur_QTseg','dur_QTc_seg',
    'dur_JT','dur_TpToff','dur_RR','dur_PR'
]
std_features = ['sd_QT','sd_RR','sd_TpToff']
dispersion_features = ['QT_dispersion','QRS_dispersion','qrs_duration']

# 7. Vectorcardiographic/angle/amplitude features as they appear in the CSV
angle_to_pctP = [
    'angle_e1_QRS_T_deg','spatial_peak_QRS_T_angle_deg',
    'spatial_mean_QRS_T_angle_deg','spatial_ventricular_activation_time_ms',
    'time_voltage_T_mean_mVms','VG_elevation_deg','ln_amp_e3_T_ln_mV',
    'norm_amp_e2_T','norm_amp_e2_QRS','T_wave_dipolar_ln_mV',
    'pct_QRS_area_RL_sagittal_percent','R_wave_amp_lead_Y_mV',
    'Q_wave_amp_lead_Z_mV','max_amp_QRS_3D_mV','dir_QRS_max_frontal_deg',
    'dir_QRS_max_sagittal_deg','azimuth_QRS_2_8_deg','ln_QRS_non_dipolar_norm',
    'QRS_T_angle_frontal_deg','QRS_T_angle_horizontal_deg',
    'ratio_ln_QRS_T_non_dipolar','max_QRS_T_angle_across_planes_deg',
    'sin_azimuth_VG_sagittal','sin_azimuth_VG_horizontal',
    'sin_max_T_angle_horizontal','VG_RVPO_mVms',
    'mean_T_minus_mean_QRS_mVms','VG_minus_T_mVms','ratio_T_max_to_mean',
    'sin_max_T_angle_sagittal','ln_amp_e2_T_ln_mV',
    'P_axis_azimuth_deg','P_axis_elevation_deg',
    'spatial_peak_P_QRS_angle_deg','spatial_mean_P_QRS_angle_deg',
    'time_voltage_P_mVms','ratio_P_max_to_mean','ln_amp_dipolar_P_ln_mV',
    'norm_amp_e2_P','ln_amp_e3_P_ln_mV','pct_P_area_RL_sagittal_percent'
]

# 8. Independent conduction disease marker
ivcd = ['IVCD']

# 9. Labels and high-level diagnosis metadata (target variables)
labels_meta = ['label', 'AI_certain', 'HCM_type', 'cardiac complications']

# Concatenate all groups into a single ordered list
new_column_order = (
    path + subject + demographics + condition_flags + base_features +
    lead_columns + duration_features + std_features + dispersion_features +
    angle_to_pctP + ivcd + labels_meta
)

# Reorder the DataFrame
df = df[new_column_order]
print(len(df.columns))
df.to_csv('full_cohort.csv', index=False)
