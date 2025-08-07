import pandas as pd
from datetime import datetime, timedelta
pd.set_option('display.max_rows', None)  
pd.set_option('display.max_columns', None)  
pd.set_option('display.width', None)  

dtype_spec = {
    'report_17': 'str',  
    'report_12': 'str',
    'report_13': 'str',
    'report_14': 'str',
    'report_15': 'str',
    'report_16': 'str'
}


ECG_df = pd.read_csv('data/machine_measurements.csv', dtype=dtype_spec)
patientdf = pd.read_csv('data/patients.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
admdf = pd.read_csv('data/admissions.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
df = pd.read_csv('data/diagnoses_icd.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
recdf = pd.read_csv('data/record_list.csv', dtype=dtype_spec)
notes = pd.read_csv('data/discharge.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')

ECG_df['ecg_time'] = pd.to_datetime(ECG_df['ecg_time']).dt.date
admdf['admittime'] = pd.to_datetime(admdf['admittime']).dt.date
admdf['dischtime'] = pd.to_datetime(admdf['dischtime']).dt.date

icd_icm = ['I255']

icd_dcm = ['I420', 'I426', '4255', '4257']                       # Dilated CM
icd_hcm = ['4251', '42511', 'I421', 'I422', '42518']            # Hypertrophic CM

# Keep only DCM + HCM rows, then flag the dilated ones
df = (
    df[df['icd_code'].isin(icd_dcm + icd_hcm + icd_icm)]     # one dataframe “on top of another”
      .copy()
)
df['Dilated'] = df['icd_code'].isin(icd_dcm + icd_icm)  # True = DCM, False = HCM
df['Ischemic'] = df['icd_code'].isin(icd_icm)  # True = ICM, False = non-ICM
# Optional sanity‑checks
print("DCM subjects :", len(df[df['Dilated']].subject_id.unique()))
print("Ischemic subjects :", len(df[df['Ischemic']].subject_id.unique()))
print("HCM subjects :", len(df[~df['Dilated']].subject_id.unique()))

dilated_lookup = (
    df[['hadm_id', 'Dilated', "Ischemic"]]
      .drop_duplicates()          # In case the same admission has multiple ICD rows
)

# 2) Keep only the CM admissions, then merge the flag in
adm_t = (
    admdf[admdf['hadm_id'].isin(dilated_lookup['hadm_id'])]
      .merge(dilated_lookup, on='hadm_id', how='left')
)

# only include admissions with ECG taken 
df = pd.merge(adm_t, ECG_df, on='subject_id', suffixes=('', '_ECG'))
df = df[(df['ecg_time'] >= df['admittime']) & (df['ecg_time'] <= df['dischtime'])]

print("DCM subjects :", len(df[df['Dilated']].subject_id.unique()))
print("HCM subjects :", len(df[~df['Dilated']].subject_id.unique()))
print("ICM subjects :", len(df[df['Ischemic']].subject_id.unique()))


df = pd.merge(df, patientdf,     left_on=['subject_id'], right_on=['subject_id'], how='left')

recdf.drop(columns=[
'file_name',
'ecg_time',
'subject_id'
], inplace=True)
df = pd.merge(df, recdf, on='study_id')



display(df.head())
#df.to_csv('full_cohort.csv')

print(len(df))
print(len(df['hadm_id'].unique()))
notes = notes[notes['hadm_id'].isin(df['hadm_id'].unique())]

notes = notes[['hadm_id', 'subject_id', 'text']].copy()
print(len(notes))
note_df = pd.merge(df[['hadm_id', 'Dilated']], notes, on=['hadm_id'], how='right')
print(len(note_df))

note_df.drop_duplicates(inplace=True)
print("Final cohort size:", len(note_df))
note_df.to_csv('discharge_notes.csv', index=False)


