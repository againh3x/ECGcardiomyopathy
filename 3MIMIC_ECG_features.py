import pandas as pd
df = pd.read_csv('full_cohort.csv')
df.drop(columns=['report_15', 'report_16', 'report_17'], inplace=True, errors='ignore')
df = df.dropna(axis=1, how='all')
report_cols = [c for c in df.columns if c.startswith("report_")]

'''this code creates new boolean columns depending on whether the select keyword appears within report_1-report_14 in the MIMIC machine measurements CSV'''


# Define the plain‑text phrases (lower‑case for easy matching)
phrases = ["abnormal ecg"]

# Build the flag
df["Abnormal ECG"] = (
    df[report_cols]
      .apply(lambda s: (
          s.str.contains(phrases[0], case=False, regex=False, na=False) 
      ))
      .any(axis=1)
)

phrases = ["borderline ecg"]

df["Borderline ECG"] = (
    df[report_cols]
      .apply(lambda s: (
          s.str.contains(phrases[0], case=False, regex=False, na=False) 
      ))
      .any(axis=1)
)

phrases = ["sinus rhythm"]

df["Sinus Rhythm"] = (
    df[report_cols]
      .apply(lambda s: (
          s.str.contains(phrases[0], case=False, regex=False, na=False)
      ))
      .any(axis=1)
)
phrases = ["Sinus br"]

df["Sinus bradycardia"] = (
    df[report_cols]
      .apply(lambda s: (
          s.str.contains(phrases[0], case=False, regex=False, na=False) 
      ))
      .any(axis=1)
)

phrases = ["sinus tac"]
df["Sinus Tachycardia"] = (
    df[report_cols]
      .apply(lambda s: (
          s.str.contains(phrases[0], case=False, regex=False, na=False)
      ))
      .any(axis=1)
)
#easier for multi-keyword search
joined_reports = (
    df[report_cols]            # slice the report columns
      .astype(str)             # make sure everything is string
      .agg(" ".join, axis=1)   # collapse to one long string
      .str.lower()             # case‑insensitive by lowering once
)

phrases = ['low voltage, precordial leads','low voltage, extremity and precordial leads','low qrs voltages in precordial leads','generalized low qrs voltages']
df["Low Precordial Voltage"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['low qrs voltages in limb leads','low voltage, extremity and precordial leads','low voltage, extremity leads','generalized low qrs voltages']
df["Low Limb Voltage"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['lvh', 'left ventricular hypertrophy']
df["LVH"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['rvh', 'right ventricular hypertrophy']
df["RVH"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['lae', 'left atrial enlargement', 'biatrial enlargement']
df["LAE"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['rae', 'right atrial enlargement', 'biatrial enlargement']
df["RAE"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['st elevation','s-t elevation', 'st-t elevation']
df["ST Elevation"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['st depression', 'st-t depression', 'st-t depression']
df["ST Depression"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['st change', 'st-t change', 'st-t change']
df["ST Changes"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['t wave changes']
df["T Wave Changes"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['atrial fib', 'a-fib', 'a fib']
df["Atrial Fibrillation"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['atrial flut', 'a-flut', 'a flut']
df["Atrial Flutter"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['lbbb', 'left bundle', 'left-bundle']
df["LBBB"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['rbbb', 'right bundle', 'right-bundle']
df["RBBB"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['IVCD', 'intraventricular conduction']
df["IVCD"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

#left anterior fascicular block
phrases = ['lafb', 'left anterior fas']
df["LAFB"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['lpfb', 'left posterior fas']
df["LPFB"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)
#left axis deviation
phrases = ['lad', 'left axis deviation', 'leftward axis']
df["LAD"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['right axis deviation', 'rightward axis']
df["RAD"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)
phrases = ['right atrial abnormality']
df["Right Atrial Abnormality"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['left atrial abnormality']
df["Left Atrial Abnormality"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases =  ['pvc', 'ventricular premature complex', 'multiple premature complexes, vent & supraven', 'premature ventricular complex', 'premature ventricular contractions', 'ventricular couplets', 'bigeminal pvcs', '- frequent premature ventricular contractions', '- premature ventricular contractions', '- ventricular couplets']
df["PVC"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)


phrases = ['rapid ventricular response', 'v-rate']
df["RVR"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)


phrases = ['slow ventricular response']
df["SVR"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)

phrases = ['infarct, recent', 'infarct, acute', ]

#remove paced ECGs
print("Original Length:" , len(df))
phrases = ['pace', 'pacing']
df["Paced ECG"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)
df = df[df['Paced ECG'] == False]
print("After removing paced ECGs:" , len(df))

#remove faulty ECGs
phrases = ['limb lead reversal', 'warning: data quality', 'unsuitable for analysis', 'data quality may affect interpretation', 'all 12 leads are missing', 'suspect arm lead reversal','recording unsuitable for analysis','please repeat', 'poor quality data', 'interpretation may be affected' ]
df["Faulty ECG"] = joined_reports.apply(
    lambda txt: any(p in txt for p in phrases)
)
df = df[df['Faulty ECG'] == False]
print("After removing low-quality ECGs:" , len(df))

df.to_csv('full_cohort.csv', index=False)
