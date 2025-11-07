# Feature engineering script for NHANES cleaned dataset
# This code will load the cleaned CSV at /mnt/data/nhanes_cleaned.csv,
# create clinically meaningful features, and save an output CSV
# to /mnt/data/nhanes_features.csv. It also prints summaries.
#
# NOTE: This runs here so you can see the resulting preview & feature list.

import pandas as pd
import numpy as np
from pathlib import Path

IN_PATH = Path("../data/processed/nhanes_cleaned.csv")
OUT_PATH = Path("../data/processed/nhanes_features.csv")

df = pd.read_csv(IN_PATH)

def first_available(df, names):
    for n in names:
        if n in df.columns:
            return n
    return None

# Map likely column names in this dataset
col_map = {
    "age": first_available(df, ["RIDAGEYR","RIDAGE"]),
    "sex": first_available(df, ["RIAGENDR","RIAGENDR"]),
    "race": first_available(df, ["RIDRETH1","RIDRETH3"]),
    "bmi": first_available(df, ["BMXBMI","BMX_BMI","BMXBMI"]),
    "waist": first_available(df, ["BMXWAIST","BMX_WAIST","BMXWAIST"]),
    "height": first_available(df, ["BMXHT","BMX_HT"]),
    "sbp": first_available(df, ["BPXSY1","BPXSY1"]),
    "dbp": first_available(df, ["BPXDI1","BPXDI1"]),
    "a1c": first_available(df, ["LBXGH","GHB","LBXGH"]),
    "glucose": first_available(df, ["LBXSGL","LBXGLU","LBXGLU","GLUCOSE","LBXSGL"]),
    "insulin": first_available(df, ["LBXINS","LBXSIR","INS","LBXIR"]),
    "tchol": first_available(df, ["LBXTC","LBXTC"]),
    "hdl": first_available(df, ["LBDHDD","LBXHDD","HDL","LBDHDD"]),
    "trig": first_available(df, ["LBXTR","TRIGLY","LBXTR","LBXTR"]),
    "creatinine": first_available(df, ["LBXSCR","URXCR","LBXSCR"]),
    "albumin": first_available(df, ["URXUMA","URXUMA"]),
    "diq": first_available(df, ["DIQ010","DIQ050","DIQ010A","DIQ010"]),
    "rx": first_available(df, ["RXDUSE","RXQ_RX","RXDUSE"]),
    "paq_cols": [c for c in df.columns if c.upper().startswith("PAQ")],
    "smq": first_available(df, ["SMQ020","SMQ020A","SMQ"]),
    "cycle": first_available(df, ["CYCLE","cycle"])
}

# Print mapping summary
print("Column mapping (first available):")
for k,v in col_map.items():
    if k=="paq_cols":
        print(f"  {k}: {len(v)} columns")
    else:
        print(f"  {k}: {v}")

# Start creating features
features = df.copy()

# Age bins
if col_map["age"]:
    features["AGE"] = features[col_map["age"]].astype(float)
    features["AGE_BIN"] = pd.cut(features["AGE"], bins=[0,18,30,45,60,75,120], labels=["child","young","adult","mid","senior","elder"])
else:
    features["AGE"] = np.nan
    features["AGE_BIN"] = np.nan

# BMI and BMI class
if col_map["bmi"]:
    features["BMI"] = features[col_map["bmi"]].astype(float)
    features["BMI_CLASS"] = pd.cut(features["BMI"], bins=[0,18.5,25,30,35,100], labels=["under","normal","over","obese1","obese2"])
else:
    features["BMI"] = np.nan
    features["BMI_CLASS"] = np.nan

# Waist-to-height ratio
if col_map["waist"] and col_map["height"]:
    # ensure height not zero / NaN
    features["WAIST_HT_RATIO"] = features[col_map["waist"]].astype(float) / features[col_map["height"]].replace({0:np.nan}).astype(float)
else:
    features["WAIST_HT_RATIO"] = np.nan

# Blood pressure: take average of up to 3 readings if present
sbp_name = col_map["sbp"]
dbp_name = col_map["dbp"]
if sbp_name:
    # there may be BPXSY1,BPXSY2,BPXSY3 columns; compute mean of those available
    sbp_cols = [c for c in df.columns if c.upper().startswith("BPXSY")]
    features["SBP_MEAN"] = features[sbp_cols].replace({0:np.nan}).astype(float).mean(axis=1)
else:
    features["SBP_MEAN"] = np.nan

if dbp_name:
    dbp_cols = [c for c in df.columns if c.upper().startswith("BPXDI")]
    features["DBP_MEAN"] = features[dbp_cols].replace({0:np.nan}).astype(float).mean(axis=1)
else:
    features["DBP_MEAN"] = np.nan

# Hypertension flag
features["HYPERTENSION_FLAG"] = ((features["SBP_MEAN"]>=130) | (features["DBP_MEAN"]>=80)).astype(int)

# Lab values: glucose, a1c, insulin
if col_map["glucose"]:
    features["GLUCOSE"] = features[col_map["glucose"]].astype(float)
else:
    features["GLUCOSE"] = np.nan

if col_map["a1c"]:
    features["A1C"] = features[col_map["a1c"]].astype(float)
else:
    features["A1C"] = np.nan

# Insulin might be missing in some cycles
if col_map["insulin"]:
    try:
        features["INSULIN"] = features[col_map["insulin"]].astype(float)
    except Exception:
        # sometimes insulin stored elsewhere
        features["INSULIN"] = pd.to_numeric(features[col_map["insulin"]], errors='coerce')
else:
    features["INSULIN"] = np.nan

# HOMA-IR if both present
features["HOMA_IR"] = np.nan
mask_homa = features["INSULIN"].notna() & features["GLUCOSE"].notna()
features.loc[mask_homa, "HOMA_IR"] = (features.loc[mask_homa, "INSULIN"] * features.loc[mask_homa, "GLUCOSE"]) / 405.0

# Lipids and ratios
if col_map["tchol"]:
    features["TCHOL"] = pd.to_numeric(features[col_map["tchol"]], errors='coerce')
else:
    features["TCHOL"] = np.nan

if col_map["hdl"]:
    features["HDL"] = pd.to_numeric(features[col_map["hdl"]], errors='coerce')
else:
    features["HDL"] = np.nan

if col_map["trig"]:
    features["TRIG"] = pd.to_numeric(features[col_map["trig"]], errors='coerce')
else:
    features["TRIG"] = np.nan

features["CHOL_HDL_RATIO"] = np.nan
mask_ch = features["TCHOL"].notna() & features["HDL"].notna()
features.loc[mask_ch, "CHOL_HDL_RATIO"] = features.loc[mask_ch, "TCHOL"] / features.loc[mask_ch, "HDL"]

features["TG_TO_HDL"] = np.nan
mask_th = features["TRIG"].notna() & features["HDL"].notna()
features.loc[mask_th, "TG_TO_HDL"] = features.loc[mask_th, "TRIG"] / features.loc[mask_th, "HDL"]

# Kidney markers: albumin/creatinine ratio if available
if col_map["albumin"] and col_map["creatinine"]:
    # albumin in URXUMA, creatinine maybe URXUCR or LBXSCR
    album_col = col_map["albumin"]
    creat_col = col_map["creatinine"]
    features["ACR"] = pd.to_numeric(features[album_col], errors='coerce') / pd.to_numeric(features[creat_col], errors='coerce')
else:
    features["ACR"] = np.nan

# Smoking flag
if col_map["smq"]:
    features["SMOKER"] = features[col_map["smq"]].apply(lambda x: 1 if str(x).strip() in ["1","1.0",1] else 0)
else:
    features["SMOKER"] = np.nan

# Physical activity score: sum available PAQ columns (numeric)
paq_cols = col_map["paq_cols"]
if paq_cols:
    # make numeric and sum (ignoring NaN)
    features["PA_SCORE"] = features[paq_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
else:
    features["PA_SCORE"] = np.nan

# Medication & self-reported diabetes flags
if col_map["diq"]:
    features["DIQ_FLAG"] = features[col_map["diq"]].apply(lambda x: 1 if str(x).strip() in ["1","1.0",1] else 0)
else:
    features["DIQ_FLAG"] = np.nan

if col_map["rx"]:
    features["ON_RX"] = features[col_map["rx"]].apply(lambda x: 1 if str(x).strip() in ["1","1.0",1] else 0)
else:
    features["ON_RX"] = np.nan

# Target creation: binary Diabetes (Diabetes if DIQ_FLAG ==1 or A1C>=6.5 or GLUCOSE>=126)
features["DIABETES_BIN"] = 0
cond = pd.Series(False, index=features.index)
if features["DIQ_FLAG"].notna().any():
    cond = cond | (features["DIQ_FLAG"]==1)
if features["A1C"].notna().any():
    cond = cond | (features["A1C"]>=6.5)
if features["GLUCOSE"].notna().any():
    cond = cond | (features["GLUCOSE"]>=126)
features.loc[cond.fillna(False), "DIABETES_BIN"] = 1

# 3-class label: 0 normal, 1 prediabetes, 2 diabetes
features["DIABETES_3CLASS"] = 0
# prediabetes: A1C 5.7-6.4 or glucose 100-125
pred_mask = ((features["A1C"]>=5.7) & (features["A1C"]<6.5)) | ((features["GLUCOSE"]>=100) & (features["GLUCOSE"]<126))
features.loc[pred_mask & (~cond), "DIABETES_3CLASS"] = 1
features.loc[cond, "DIABETES_3CLASS"] = 2

# Select a compact set of model-ready features to output
model_features = [
    "SEQN","CYCLE","AGE","AGE_BIN","RIAGENDR","RIDRETH1","BMI","BMI_CLASS","WAIST_HT_RATIO",
    "SBP_MEAN","DBP_MEAN","HYPERTENSION_FLAG","GLUCOSE","A1C","INSULIN","HOMA_IR",
    "TCHOL","HDL","TRIG","CHOL_HDL_RATIO","TG_TO_HDL","ACR","SMOKER","PA_SCORE",
    "DIQ_FLAG","ON_RX","DIABETES_BIN","DIABETES_3CLASS"
]

# Keep only those that exist in features
model_features = [f for f in model_features if f in features.columns]
out = features[model_features].copy()

# Convert categorical bins to strings (for safer saving)
if "AGE_BIN" in out.columns:
    out["AGE_BIN"] = out["AGE_BIN"].astype(str)
if "BMI_CLASS" in out.columns:
    out["BMI_CLASS"] = out["BMI_CLASS"].astype(str)

# Save output
out.to_csv(OUT_PATH, index=False)
print(f"Saved engineered features to: {OUT_PATH}")
print("Output shape:", out.shape)
print("\nColumns saved:")
print(out.columns.tolist())

# Show head and basic stats
print(out.head())
print(out.describe(include='all').T.head(40))