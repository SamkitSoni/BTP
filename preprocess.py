import pandas as pd
import numpy as np
import os
import json
import re

RAW = "data/diabetic_data.csv"
CLEAN = "data/diabetic_clean.csv"
FEATURES_JSON = "data/features.json"

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses"
]

CAT_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "max_glu_serum", "A1Cresult",
    "metformin", "insulin", "change", "diabetesMed"
]

def _sanitize(name: str):
    return re.sub(r"[^\w]+", "_", name).strip("_")

def parse_age(s):
    """Convert [30-40) to 35"""
    try:
        s = s.replace("[", "").replace(")", "").replace("]", "")
        a, b = s.split("-")
        return (int(a) + int(b)) / 2
    except:
        return np.nan

def load_and_clean(inpath=RAW, outpath=CLEAN, features_json=FEATURES_JSON):

    print(f"[INFO] Loading raw file: {inpath}")
    df = pd.read_csv(inpath)

    # Remove unwanted IDs
    for c in ("encounter_id", "patient_nbr"):
        if c in df:
            df = df.drop(columns=c)

    # Create binary label
    df["readmit_30"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])

    # Hospital ID
    df["hospitalid"] = df.get("admission_source_id", 0)

    # Only keep relevant columns
    keep = NUMERIC_COLS + CAT_COLS + ["hospitalid", "readmit_30"]
    keep = [c for c in keep if c in df]
    df = df[keep].copy()

    # Clean numeric columns
    for c in NUMERIC_COLS:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())

    # Clean age
    if "age" in df:
        df["age"] = df["age"].astype(str).apply(parse_age)
        df["age"] = df["age"].fillna(df["age"].median())

    # One-hot encode ALL categorical columns EXCEPT age  
    cats = [c for c in CAT_COLS if c in df and c != "age"]
    if cats:
        df_cat = df[cats].fillna("NA").astype(str)
        ohe = pd.get_dummies(df_cat, prefix=cats)
        ohe.columns = [_sanitize(c) for c in ohe.columns]
        df = pd.concat([df.drop(columns=cats), ohe], axis=1)

    # Sanitize column names
    df.columns = [_sanitize(c) for c in df.columns]

    # Convert all non-label columns to numeric
    for col in df.columns:
        if col != "readmit_30":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Final fill
    df = df.fillna(df.median(numeric_only=True))

    # Save cleaned CSV
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    df.to_csv(outpath, index=False)
    print(f"[INFO] Saved cleaned CSV → {outpath}, shape={df.shape}")

    # Build final feature list (NO categorical raw columns!)
    features = [
        c for c in df.columns
        if c not in ("readmit_30", "hospitalid")  
        and not c.startswith("race$")  
        and c != "race"                   # IMPORTANT
    ]

    # Write features.json
    with open(features_json, "w") as f:
        json.dump(features, f, indent=2)

    print(f"[INFO] Saved {len(features)} features → {features_json}")
    print("[INFO] First 15 features:", features[:15])

    return outpath, features


if __name__ == "__main__":
    load_and_clean()
