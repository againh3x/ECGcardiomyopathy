import pandas as pd
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import List

df = pd.read_csv('full_cohort.csv')
df = df[df['AI_certain'] == True]
df = df[df['cardiac complications'] == False]
df = df[df['label'] == 'Dilated Ischemic']
subjects = df['hadm_id'].unique()
client = OpenAI(api_key="placeholder key")

# ---------------------------------------------------------------------#
# 2.  Pydantic schema for the structured answer
# ---------------------------------------------------------------------#
class CMClassification(BaseModel):

    dilated: bool = Field(
        description="True if the patient's ischemic cardiomyopathy is also verified to be dilated."
    )

BASE_PROMPT = """
This patient was marked down for ischemic cardiomyopathy. However, it may or may not be dilated. The discharge note will be provided below. Output True or False for whether or not the patient also has dilated cardiomyopathy.
"""
def classify_note(note_text: str) -> CMClassification:

    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": BASE_PROMPT},
            {"role": "user", "content": note_text},
        ],
        response_format=CMClassification,

    )

    return response.choices[0].message.parsed  # → CMClassification instance

IN_CSV  = "discharge_notes.csv"  # Path to the input CSV file
OUT_CSV = "classified_ischemic.csv"

df = pd.read_csv(IN_CSV)
df = df[df['hadm_id'].isin(subjects)]  # Filter to only include relevant subjects
df = df[['hadm_id', 'text']]  # Keep only necessary columns
results: List[dict] = []

for idx, row in df.iterrows():
    hadm_id   = row["hadm_id"]
    note_text = str(row["text"])
   

    
    parsed: CMClassification = classify_note(note_text)
    payload = parsed.model_dump()
    


    results.append({"hadm_id": hadm_id, **payload})
    print(results[-1])

    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(df)}")

pd.DataFrame(results).to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved results → {OUT_CSV}")


idf = pd.read_csv('classified_ischemic.csv')
fdf = pd.read_csv('full_cohorted.csv')
idf = idf[idf['dilated'] == False]
fdf = fdf[fdf['hadm_id'].isin(idf['hadm_id'].unique())]

df = pd.read_csv('table_data.csv')
df = df[df['AI_certain'] == True]
df = df[df['cardiac complications'] == False]
print(len(df))
print(len(idf))

df = df[~((df['path'].isin(fdf['path'].unique())) & (df['label'] == 'Dilated Ischemic'))]
df = df[~(df['label'] == 'Dilated Unknown')]

print(len(df))
df.to_csv('full_cohort.csv', index=False)
