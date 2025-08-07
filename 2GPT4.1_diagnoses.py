"""
Classify cardiomyopathy features in discharge notes.

• Reads /home/oai/share/discharge_notes.csv  (needs columns: hadm_id, text, Dilated)
• Writes /home/oai/share/classified_discharge_notes.csv

Replace "YOUR_OPENAI_KEY" below with the real key before running.
"""

from typing import List, Optional

import os
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ---------------------------------------------------------------------#
# 1.  OpenAI client (HARD‑CODED KEY – replace before use)
# ---------------------------------------------------------------------#
client = OpenAI(api_key="placeholder_key")

# ---------------------------------------------------------------------#
# 2.  Pydantic schema for the structured answer
# ---------------------------------------------------------------------#
class CMClassification(BaseModel):
    cardiomyopathy: str = Field(
        description='Exactly one of "Hypertrophic", "Dilated", "Neither"'
    )
    phenotype: str = Field(
        description=(
            'For HCM: "septal" | "apical" | "unknown";  '
            'For DCM: "ischemic" | "non-ischemic" | "unknown"'
        )
    )
    obstructive: Optional[str] = Field(
        default=None,
        description=(
            '"Obstructive" | "Non-obstructive" | '
            '"Probably non-obstructive" | "Unknown"  (HCM only)'
            '''“Obstructive” if LVOT gradient ≥30 mmHg at rest or clearly stated obstructive HCM or SAM + gradient.
“Non-obstructive” if echo says no obstruction / LVOT gradient <30 or explicitly “non-obstructive.”
“Probably non-obstructive” if echo likely shows no obstruction but wording is indirect.
“Unknown” if no echo or no info.'''


        ),
    )
    surgery: bool = Field(
        description="True if septal myectomy / EtOH ablation during THIS stay"
    )

# ---------------------------------------------------------------------#
# 3.  Prompt helpers
# ---------------------------------------------------------------------#
BASE_PROMPT = """
I will give you a discharge note from the hospital of this patient who was marked down for {extra} cardiomyopathy. Inspect it very carefully. The first word you return should be one of the only three options: “Hypertrophic” “Dilated” (includes Ischemic) or “Neither”. If the discharge note mentions that the patient has hypertrophic or dilated (any form of either) cardiomyopathy, then that should be the value. If there are truly zero mentions to either, it should be neither.

Now, the next value is phenotype. If the patient has hypertrophic cardiomyopathy, you will enter one of the following phenotypes: septal, apical, or unknown. If the discharge note (indirectly or directly) mentions that the hypertrophic cardiomyopathy is septal, you can mark that down for phenotype. If it is apical, likewise. If there is truly no mention or an echo was not taken in the admission to decide, put unknown. 

For DCM, if Dilated is the value for cardiomyopathy, the only types I want you to consider is ischemic and non-ischemic dilated cardiomyopathy. Look for mentions in the text for each person and if Dilated is the value for CM, the phenotype should say “ischemic”, “non-ischemic”, or unknown.

Now finally, the second last thing I want you to add and fill out is “Obstructive?” If the patient had an echocardiogram taken during the admission and the LVOT was obstructive, or if it is known that the HCM is obstructive, the Obstructive? value should be “Obstructive”. If it is known that the HCM is non obstructive (from an echo etc.), the value should be “Non-obstructive”. If it is truly unknown at all, “Unknown”, and if it can be inferred that it is probably non obstructive but the text doesn't fully state it or not or state the LVOT level, “Probably non-obstructive”. This only applies for HCM, DCM patients can just have their value nan. 

Finally, the last thing I want you to fill out (again only applies for HCM) is “Surgery”. If the note specifically mentions that the patient had a septal alcohol ablation or a myectomy to reduce obstruction DURING THE STAY OF THE NOTE, the value should be True. This ONLY applies if they had the surgery or procedure during THIS stay, NOT in the past (if in the past then False). For all DCM this can be False. 

"""

def build_prompt(cm_hint: str) -> str:
    extra = (
        f"  The patient was pre‑tagged as {cm_hint} cardiomyopathy."
        if cm_hint in {"dilated", "hypertrophic"}
        else ""
    )
    return BASE_PROMPT.format(extra=extra)

# ---------------------------------------------------------------------#
# 4.  Single‑note call using the Pydantic schema
# ---------------------------------------------------------------------#
def classify_note(note_text: str, cm_hint: str) -> CMClassification:
    prompt = build_prompt(cm_hint)

    response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": note_text},
        ],
        response_format=CMClassification,

    )

    return response.choices[0].message.parsed  # → CMClassification instance

# ---------------------------------------------------------------------#
# 5.  CSV driver
# ---------------------------------------------------------------------#
def main() -> None:
    IN_CSV  = "discharge_notes.csv"  # Path to the input CSV file
    OUT_CSV = "classified_hadm.csv"

    df = pd.read_csv(IN_CSV)
    results: List[dict] = []

    for idx, row in df.iterrows():
        hadm_id   = row["hadm_id"]
        note_text = str(row["text"])
        cm_hint   = "dilated" if bool(row["Dilated"]) else "hypertrophic"

     
        parsed: CMClassification = classify_note(note_text, cm_hint)
        payload = parsed.model_dump()
       


        results.append({"hadm_id": hadm_id, **payload})
        print(results[-1])

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)}")

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\n✅ Saved results → {OUT_CSV}")

# ---------------------------------------------------------------------#
if __name__ == "__main__":
    main()
