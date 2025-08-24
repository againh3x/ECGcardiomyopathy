#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Three-stage LLM pipeline for cardiomyopathy labeling from discharge notes.

Stage 1 (all admissions):
  → CM_label in {"HCM","DCM","Neither"}
  → EF (float or None)
  → q_cardiomyopathy (list of short quotes)

Stage 2 (if HCM):
  → Obstruction in {"True","False","Unknown"} for THIS admission (echo/LVOT gradient)
  → Surgery in {True, False}  (True only if myectomy or septal alcohol ablation DURING THIS admission)
  → q_obstruction (list of short quotes)

Stage 3 (if DCM):
  → Ischemic in {"True","False","Unknown"} (CAD/MI/revascularization/RWMA/explicit text)
  → q_ischemic (list of short quotes)

Output CSV columns (one row per hadm_id):
  CM_label, Obstruction, Ischemic, EF, Surgery, q_cardiomyopathy, q_obstruction, q_ischemic

Input:  discharge_notes.csv  with columns: hadm_id, text
Output: cm_pipeline_labels.csv
"""

from __future__ import annotations

import os
import time
import random
from typing import Optional, List, Literal

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI

# ───────────────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = "" #placeholder      
MODEL_NAME = "gpt-4.1"
TEMPERATURE = 0

MAX_RETRIES = 3
BASE_BACKOFF = 2.0   # seconds (exponential with jitter)

IN_CSV = "discharge_notes.csv"
OUT_CSV = "cm_pipeline_labels.csv"

client = OpenAI(api_key=OPENAI_API_KEY)

# ───────────────────────────────────────────────────────────────────────────────
# Stage 1 schema: CM label + EF + supporting quotes
# ───────────────────────────────────────────────────────────────────────────────
class Stage1Result(BaseModel):
    CM_label: Literal["HCM", "DCM", "Neither"] = Field(
        description="Return 'HCM' only if the note clearly indicates hypertrophic cardiomyopathy (includes echo findings or the patient's PMH); "
                    "Return 'DCM' only if clearly dilated CM or on the echocardiogram BOTH LV dilation and reduced EF are documented; "
                    "Else return 'Neither'."
    )
    EF: Optional[float] = Field(
        default=None,
        description="LVEF % as a number if reported (e.g., 25, 45, 60). Include the number from the echocardiogram taken within the the CURRENT visit. None if not clearly stated."
    )
    q_cardiomyopathy: List[str] = Field(
        default_factory=list,
        description="1–5 short verbatim quotes that support the CM_label decision (e.g., 'LV moderately dilated', 'EF 25%')."
    )

STAGE1_SYSTEM = """
You are a careful clinical information extractor. Work ONLY from the discharge note text.
Task: Decide if the admission clearly has HCM, DCM, or Neither.
Rules:
- HCM: text clearly indicates hypertrophic cardiomyopathy (current admission or clear history) or echocardiogram is clearly indicative of HCM.
- DCM: ONLY if clearly dilated cardiomyopathy (includes both ischemic DILATED and non-ischemic DILATED cardiomyopathy) or BOTH are documented: (1) LV dilation/enlargement and (2) reduced EF. If either is missing/unclear, do NOT choose DCM. Ischemic cardiomyopathy without LV dilation is NOT DCM.
- Neither: if neither HCM nor DCM is clearly supported by the note.

Extract EF as a numeric percent when stated (e.g., 'EF 25% -> 25'); otherwise EF=None.
Return 1–5 short verbatim quotes in q_cardiomyopathy that justify CM_label.
Do not infer beyond the text. Be conservative.
"""

STAGE1_USER = "Classify cardiomyopathy for THIS admission and extract EF from the following discharge note:"

# Stage 2 (HCM-only): obstruction + surgery + quotes

class Stage2HCMResult(BaseModel):
    Obstruction: Literal["True", "False", "Unknown"] = Field(
        description="For THIS admission, based on explicit echo/LVOT documentation: "
                    "'True' if obstructive LVOT gradient ≥30 mmHg (make sure it is LVOT) OR text clearly states any form of obstruction from the echocardiogram taken during the visit OR clear past of HOCM and NO recent contradictory non-obstructive echo findings; "
                    "'False' if text clearly indicates no LVOT obstruction or LVOT on current echo <30 mmHg (make sure it is LVOT and not another gradient); "
                    "'Unknown' if unclear or no echo this admission."
    )
    Surgery: bool = Field(
        description="True ONLY if septal myectomy or alcohol septal ablation clearly occurred DURING THIS admission. "
                    "If not mentioned or historical only, return False."
    )
    q_obstruction: List[str] = Field(
        default_factory=list,
        description="1–5 short verbatim quotes supporting the obstruction decision (e.g., 'LVOT gradient 64 mmHg', 'No obstruction', 'mild LVOT obstruction')."
    )

STAGE2_HCM_SYSTEM = """
You are an information extractor for HCM admissions. Work ONLY from the discharge note text for THIS admission. The patient has hypertrophic cardiomyopathy.
Decide LVOT obstruction status and whether a septal reduction procedure occurred during THIS admission.
Rules:
- Obstruction=True if LVOT gradient ≥30 mmHg (rest or provoked) or any form of obstruction is noted from the echocardiogram conducted in the visit. recent obstruction in the past is also enough for HOCM (as long as the most recent or current echo is >30 mmHg LVOT gradient).
- Obstruction=False if text explicitly states non-obstructive HCM or LVOT gradient reported is <30 mmHg.
- Obstruction=Unknown if no clear echo/LVOT evidence this admission.
- Surgery=True ONLY if septal myectomy or alcohol septal ablation (ASA) occurred DURING THIS admission; else False.
Return 1–5 short quotes in q_obstruction that justify the obstruction decision.
"""

STAGE2_HCM_USER = "For this HCM admission, determine obstruction and surgery status from the discharge note:"

# Stage 3 (DCM-only): ischemic status + quotes

class Stage3DCMResult(BaseModel):
    Ischemic: Literal["True", "False", "Unknown"] = Field(
        description="For THIS admission, is the DCM ischemic? "
                    "True if any ischemic evidence is documented (CAD/prior MI, CABG/PCI/stents, RWMA consistent with ischemia, "
                    "or explicit 'ischemic cardiomyopathy'); "
                    "False if there is clear non-ischemic/idiopathic/viral/toxin documentation OR no ischemic evidence; "
                    "Unknown only if the note is too ambiguous to tell."
    )
    q_ischemic: List[str] = Field(
        default_factory=list,
        description="1–5 short verbatim quotes supporting the ischemic decision (e.g., 'prior MI', 'CABG in 2012', 'No CAD')."
    )

STAGE3_DCM_SYSTEM = """
You are an information extractor for DCM admissions. Work ONLY from the discharge note text for THIS admission.
Decide if the dilated cardiomyopathy is ischemic or not.
Rules:
- Ischemic=True if ANY of the following appear: CAD/prior MI documented; revascularization (CABG/PCI/stents); RWMA consistent with ischemia; explicit 'ischemic cardiomyopathy'.
- Ischemic=False if the note is clearly non-ischemic/idiopathic/viral/toxin OR there is no ischemic evidence and is reasonably not ischemic.
- Ischemic=Unknown if the text is ambiguous and a decision between ischemic or not ischemic is unknown.
Return 1–5 short quotes in q_ischemic that justify your decision.
"""

STAGE3_DCM_USER = "For this DCM admission, decide if it is ischemic vs non-ischemic from the discharge note:"

#caller
def call_structured(schema_cls, system_prompt: str, user_prefix: str, note_text: str):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.beta.chat.completions.parse(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"{user_prefix}\n\n{note_text}"},
                ],
                response_format=schema_cls,
            )
            return resp.choices[0].message.parsed
        except ValidationError as ve:
            if attempt == MAX_RETRIES:
                raise
            time.sleep((BASE_BACKOFF ** attempt) * (1 + random.random()))
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            time.sleep((BASE_BACKOFF ** attempt) * (1 + random.random()))
    raise RuntimeError("LLM structured call failed after retries.")

# main

df = pd.read_csv(IN_CSV)
if not {"hadm_id", "text"}.issubset(df.columns):
    raise ValueError("Input CSV must contain columns: hadm_id, text")

rows = []
total = len(df)

for i, r in df.iterrows():
    hadm_id = r["hadm_id"]
    note = str(r["text"])

    # Stage 1: CM label + EF + quotes
    s1: Stage1Result = call_structured(Stage1Result, STAGE1_SYSTEM, STAGE1_USER, note)
    cm_label = s1.CM_label
    ef_value = s1.EF
    q_cm = s1.q_cardiomyopathy or []

    obstruction = ""
    surgery = False
    q_obstruction = []
    ischemic = ""
    q_ischemic = []
    
    # Stage 2: HCM subtyping
    if cm_label == "HCM":
        s2: Stage2HCMResult = call_structured(Stage2HCMResult, STAGE2_HCM_SYSTEM, STAGE2_HCM_USER, note)
        obstruction = s2.Obstruction            # "True" | "False" | "Unknown"
        surgery = bool(s2.Surgery)              # True if procedure DURING this admission
        q_obstruction = s2.q_obstruction or []

    # Stage 3: DCM subtyping
    elif cm_label == "DCM":
        s3: Stage3DCMResult = call_structured(Stage3DCMResult, STAGE3_DCM_SYSTEM, STAGE3_DCM_USER, note)
        ischemic = s3.Ischemic                  # "True" | "False" | "Unknown"
        q_ischemic = s3.q_ischemic or []

    if cm_label == "HCM":
        if obstruction == "True":
            final_lbl = "HOCM"
        elif obstruction == "False":
            final_lbl = "HNCM"
        else:
            final_lbl = "HCM unknown"
    elif cm_label == "DCM":
        if ischemic == "True":
            final_lbl = "DCM-I"
        elif ischemic == "False":
            final_lbl = "DCM-NI"
        else:
            final_lbl = "DCM unknown"
    else:
        final_lbl = "neither"
    # Build row (quotes joined with | to fit CSV)
    row = {
        "hadm_id": hadm_id,
        "CM_label": cm_label,
        "Obstruction": obstruction,                          # "", "True", "False", "Unknown"
        "Ischemic": ischemic,                                # "", "True", "False", "Unknown"
        "EF": ef_value,                                      # float or None
        "Surgery": surgery,                                  # bool
        "q_cardiomyopathy": " | ".join(q_cm),
        "q_obstruction": " | ".join(q_obstruction),
        "q_ischemic": " | ".join(q_ischemic),
        "Final_Label": final_lbl 
    }

    print(f"{i+1}/{total}  hadm_id={hadm_id}  final_label={final_lbl}")

    rows.append(row)

    if (i + 1) % 25 == 0:
        print(f"Processed {i+1}/{total}")

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved results → {OUT_CSV}")

