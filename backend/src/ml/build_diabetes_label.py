# backend/src/ml/build_diabetes_label.py
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "src" / "data" / "processed" / "nhanes_raw_merged2.csv"  # CHANGE if needed
OUT_PATH = ROOT / "src" / "data" / "processed" / "nhanes_master_labeled.csv"

print("Loading raw merged NHANES (this may take a minute)...")
df = pd.read_csv(RAW_PATH)

cols = {c.upper(): c for c in df.columns}  # map uppercase -> original

def has(*names):
    for n in names:
        if n.upper() in cols:
            return cols[n.upper()]
    return None

# 1) diagnosis flag from questionnaire
diq_col = has("DIQ010", "DIQ010A", "DIQ010B")
if diq_col:
    df["DIAG_FLAG"] = (df[diq_col] == 1).astype(int)
else:
    df["DIAG_FLAG"] = 0
print("DIQ col used:", diq_col)

# 2) A1C (glycohemoglobin) variants
a1c_col = has("LBXGH", "LBXA1C", "LBXGH_D")
if a1c_col:
    df["A1C_VAL"] = pd.to_numeric(df[a1c_col], errors="coerce")
    df["HIGH_A1C"] = (df["A1C_VAL"] >= 6.5).astype(int)
else:
    df["HIGH_A1C"] = 0
print("A1C col used:", a1c_col)

# 3) Glucose fasting variants (mg/dL)
glu_col = has("LBXGLU", "LBXSGL", "LBXGLU_D")
if glu_col:
    df["GLU_VAL"] = pd.to_numeric(df[glu_col], errors="coerce")
    df["HIGH_GLU"] = (df["GLU_VAL"] >= 126).astype(int)
else:
    df["HIGH_GLU"] = 0
print("Glucose col used:", glu_col)

# 4) Insulin (optional)
ins_col = has("LBXIN", "LBXSIR")
print("Insulin col used:", ins_col)

# 5) Medication flags - try RXDDRUG or RXDUSE / RXDDRGID
rxdrug_col = has("RXDDRUG", "RXDRUG", "RXDUSE", "RXDDRGID", "RXDDRUG_NAME")
med_flag = pd.Series(0, index=df.index)

if rxdrug_col:
    # search for common diabetes drug keywords in drug name strings
    keywords = ["metformin","glipizide","glyburide","pioglitazone","rosiglitazone",
                "sitagliptin","linagliptin","saxagliptin","alogliptin",
                "liraglutide","semaglutide","empagliflozin","dapagliflozin",
                "insulin","canagliflozin","vildagliptin","repaglinide","nateglinide"]
    # convert to string and lowercase
    med_series = df[rxdrug_col].astype(str).str.lower().fillna("")
    pat = re.compile("|".join([re.escape(k) for k in keywords]))
    med_flag = med_series.str.contains(pat).astype(int)
else:
    # try RXDUSE (binary)
    rxuse = has("RXDUSE",)
    if rxuse:
        med_flag = (pd.to_numeric(df[rxuse], errors="coerce") == 1).astype(int)

df["MED_FLAG"] = med_flag
print("RX column used for med detection:", rxdrug_col or rxuse)

# 6) Compose final DIABETES_BIN
df["DIABETES_BIN"] = ((df["DIAG_FLAG"] == 1) | (df["HIGH_A1C"] == 1) | (df["HIGH_GLU"] == 1) | (df["MED_FLAG"] == 1)).astype(int)

# Report counts then save
print("DIABETES_BIN distribution:")
print(df["DIABETES_BIN"].value_counts(dropna=False))

# Save only necessary columns + SEQN to keep file small
cols_to_save = ["SEQN","DIABETES_BIN"]
# if SEQN absent try alternate id names
if "SEQN" not in df.columns:
    # find numeric id column
    ids = [c for c in df.columns if c.upper().startswith("SEQN") or c.upper().endswith("SEQ")]
    if ids:
        cols_to_save[0] = ids[0]
    else:
        raise RuntimeError("No SEQN-like ID column found. Add SEQN column before running.")

df[cols_to_save].to_csv(OUT_PATH, index=False)
print("Saved labeled master to:", OUT_PATH)
