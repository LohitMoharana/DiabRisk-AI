# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from lightgbm import LGBMClassifier
# from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
# from sklearn.impute import SimpleImputer
# import joblib, os
#
# RAW_PATH = "../data/processed/nhanes_raw_merged2.csv"
# SAVE_DIR = "../models/"
# os.makedirs(SAVE_DIR, exist_ok=True)
#
# print("Loading already merged dataset...")
# df = pd.read_csv(RAW_PATH)
# print("Shape:", df.shape)
#
# # Keep 2011–2018 cycles only
# valid_cycles = ["2011_2012", "2013_2014", "2015_2016", "2017_2018"]
# df = df[df["cycle"].isin(valid_cycles)]
# print("After cycle filter:", df.shape)
#
# # ------------------------------------------
# # Standardize Glucose
# # ------------------------------------------
# mgdl_cols = [c for c in df.columns if "glu" in c.lower() and "mmol" not in c.lower()]
# mmol_cols = [c for c in df.columns if "glu" in c.lower() and "mmol" in c.lower()]
#
# def pick(col_list):
#     return col_list[0] if len(col_list) > 0 else None
#
# mgdl_col = pick(mgdl_cols)
# mmol_col = pick(mmol_cols)
#
# df["GLU_VAL"] = np.nan
#
# if mmol_col:
#     df["GLU_VAL"] = df[mmol_col].astype(float) * 18
#
# if mgdl_col:
#     df.loc[df[mgdl_col].notna(), "GLU_VAL"] = df[mgdl_col].astype(float)
#
# df = df[df["GLU_VAL"].notna()]
# print("After glucose processing:", df.shape)
#
# # ------------------------------------------
# # Create label: FPG ≥ 126 mg/dL = diabetes
# # ------------------------------------------
# df["diabetes_label"] = (df["GLU_VAL"] >= 126).astype(int)
# print(df["diabetes_label"].value_counts())
#
# # ------------------------------------------
# # Temporal split (No leakage)
# # ------------------------------------------
# train_cycles = ["2011_2012", "2013_2014", "2015_2016"]
# test_cycle   = ["2017_2018"]
#
# train_df = df[df["cycle"].isin(train_cycles)]
# test_df  = df[df["cycle"].isin(test_cycle)]
#
# train_df, val_df = train_test_split(
#     train_df, test_size=0.2, random_state=42,
#     stratify=train_df["diabetes_label"]
# )
#
# print("Train / Val / Test sizes:", len(train_df), len(val_df), len(test_df))
#
# y_train = train_df["diabetes_label"]
# y_val   = val_df["diabetes_label"]
# y_test  = test_df["diabetes_label"]
#
# X_train = train_df.drop(["diabetes_label", "cycle"], axis=1)
# X_val   = val_df.drop(["diabetes_label", "cycle"], axis=1)
# X_test  = test_df.drop(["diabetes_label", "cycle"], axis=1)
#
# # ------------------------------------------
# # Impute missing
# # ------------------------------------------
# imputer = SimpleImputer(strategy="median")
# X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
# X_val   = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
# X_test  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
#
# # ------------------------------------------
# # Train high-accuracy model
# # ------------------------------------------
# model = LGBMClassifier(
#     n_estimators=2000,
#     learning_rate=0.01,
#     num_leaves=64,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     objective="binary",
#     random_state=42
# )
#
# print("Training LightGBM...")
# model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=200)
#
# # ------------------------------------------
# # Test eval
# # ------------------------------------------
# preds = model.predict(X_test)
# probs = model.predict_proba(X_test)[:, 1]
#
# print("\nResults on 2017-2018 held-out set:")
# print("Accuracy:", accuracy_score(y_test, preds))
# print("ROC-AUC:", roc_auc_score(y_test, probs))
# print("Confusion:\n", confusion_matrix(y_test, preds))
# print("\nReport:\n", classification_report(y_test, preds))
#
# # ------------------------------------------
# # Save model + imputer
# # ------------------------------------------
# joblib.dump(model, f"{SAVE_DIR}/diabetes_model.pkl")
# joblib.dump(imputer, f"{SAVE_DIR}/imputer.pkl")
#
# print("✅ Saved: model + imputer")


# Retry optimized cleaning + feature engineering (vectorized replacements, selective numeric conversion)
import pandas as pd, numpy as np
from pathlib import Path
OUT_DIR = Path("../data/diabrisk_outputs")
OUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv("../data/processed/nhanes_raw_merged2.csv", low_memory=False)
print("Loaded:", df.shape)

# Resolve cycle (CYCLE exists)
if "CYCLE" in df.columns:
    df["cycle"] = df["CYCLE"].astype(str)
elif "SDDSRVYR" in df.columns:
    cycle_map = {7: "2011_2012", 8: "2013_2014", 9: "2015_2016", 10: "2017_2018"}
    df["cycle"] = df["SDDSRVYR"].map(cycle_map)
else:
    raise RuntimeError("No cycle column found.")

valid_cycles = ["2011_2012","2013_2014","2015_2016","2017_2018"]
df = df[df["cycle"].isin(valid_cycles)].copy()
print("After cycle filter:", df.shape)

# Drop duplicates by SEQN
if "SEQN" in df.columns:
    before = df.shape[0]
    df = df.drop_duplicates(subset=["SEQN"], keep="first")
    print(f"Dropped duplicates by SEQN: {before - df.shape[0]} rows")

# Vectorized replacement of common skip strings
skip_strs = ["", " ", "NA", "N/A", "nan", "None",
             "999999","99999","9999","999","99","9",
             "777777","77777","7777","777","77","7"]
df.replace(skip_strs, np.nan, inplace=True)

# Numeric skip codes replace
numeric_skips = [7,9,77,99,777,999,7777,9999,77777,99999,777777,999999]
df.replace(numeric_skips, np.nan, inplace=True)

# Identify glucose source efficiently
chosen = None
for cand in ["LBXGLU","LBXSGL","LBXGLU","LBXGLU"]:
    if cand in df.columns:
        df["GLU_VAL"] = pd.to_numeric(df[cand], errors="coerce")
        chosen = cand
        break
# fallback to SI
if chosen is None:
    for si in ["LBDGLUSI","LBDSGLSI","LBDSGLSI"]:
        if si in df.columns:
            df["GLU_VAL"] = pd.to_numeric(df[si], errors="coerce") * 18.0182
            chosen = si
            break

print("Chosen glucose column:", chosen)
print("GLU_VAL non-null:", int(df["GLU_VAL"].notna().sum()))

# HbA1c
df["A1C"] = pd.to_numeric(df["LBXGH"], errors="coerce") if "LBXGH" in df.columns else np.nan
print("A1C non-null:", int(df["A1C"].notna().sum()))

# Preserve or derive DIABETES_BIN
if "DIABETES_BIN" in df.columns:
    df["DIABETES_BIN"] = pd.to_numeric(df["DIABETES_BIN"], errors="coerce")
else:
    df["DIABETES_BIN"] = ((df["GLU_VAL"] >= 126) | (df["A1C"] >= 6.5)).astype(int)

print("DIABETES_BIN value counts:")
print(df["DIABETES_BIN"].value_counts(dropna=False).to_dict())

# Drop leakage columns (explicit list + DIQ175*)
to_drop = ["DIQ010","DIQ050","DIQ070","DIQ160","DIQ180","DIQ230"]
to_drop += [c for c in df.columns if str(c).startswith("DIQ175")]
to_drop = [c for c in to_drop if c in df.columns]
print("Dropping leakage columns:", to_drop[:20])
df.drop(columns=to_drop, inplace=True, errors="ignore")

# Now selective numeric conversion for features used in engineered set
def to_num(col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    else:
        return pd.Series(np.nan, index=df.index)

df["BMI"] = to_num("BMXBMI")
df["AGE"] = to_num("RIDAGEYR")
df["SEX"] = df["RIAGENDR"] if "RIAGENDR" in df.columns else np.nan
df["WAIST"] = to_num("BMXWAIST")
df["HEIGHT"] = to_num("BMXHT")
df["WAIST_HT_RATIO"] = df["WAIST"] / df["HEIGHT"]

# BP means
sbp_cols = [c for c in ["BPXSY1","BPXSY2","BPXSY3","BPXSY4"] if c in df.columns]
dbp_cols = [c for c in ["BPXDI1","BPXDI2","BPXDI3","BPXDI4"] if c in df.columns]
if sbp_cols:
    df["SBP_MEAN"] = df[sbp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
else:
    df["SBP_MEAN"] = np.nan
if dbp_cols:
    df["DBP_MEAN"] = df[dbp_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
else:
    df["DBP_MEAN"] = np.nan
df["HYPERTENSION_FLAG"] = ((df["SBP_MEAN"] >= 130) | (df["DBP_MEAN"] >= 80)).astype(int)

# Lipids
df["TCHOL"] = to_num("LBXTC")
# LBDHDD or LBDHDD present
hdl_col = None
for c in ["LBDHDD","LBDHDD","LBXHDD","LBDHDD"]:
    if c in df.columns:
        hdl_col = c
        break
if hdl_col:
    df["HDL"] = to_num(hdl_col)
else:
    # try generic HDL-like columns
    hdl_cols = [c for c in df.columns if "HDL" in str(c).upper()]
    df["HDL"] = to_num(hdl_cols[0]) if hdl_cols else np.nan

df["TRIG"] = to_num("LBXTR")

df["CHOL_HDL_RATIO"] = df["TCHOL"] / df["HDL"]
df["TG_TO_HDL"] = df["TRIG"] / df["HDL"]

# Insulin and HOMA-IR
df["INSULIN"] = to_num("LBXIN")
df["HOMA_IR"] = np.where(df["INSULIN"].notna() & df["GLU_VAL"].notna(),
                         (df["INSULIN"] * df["GLU_VAL"]) / 405.0, np.nan)

# ACR: URXUMA / URXUCR
df["ACR"] = np.where(df["URXUMA"].notna() & df["URXUCR"].notna(),
                     to_num("URXUMA") / to_num("URXUCR"), np.nan)

# Smoking flag
df["SMOKER"] = to_num("SMQ020")

# PA score: sum of PAQ* numeric columns (select present PAQ columns)
paq_cols = [c for c in df.columns if str(c).upper().startswith("PAQ")]
if paq_cols:
    df["PA_SCORE"] = df[paq_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)
else:
    df["PA_SCORE"] = np.nan

# BMI class and age bin
def bmi_class(b):
    if pd.isna(b): return np.nan
    if b < 18.5: return "underweight"
    if b < 25: return "normal"
    if b < 30: return "overweight"
    return "obese"

df["BMI_CLASS"] = df["BMI"].apply(bmi_class)
df["AGE_BIN"] = pd.cut(df["AGE"], bins=[0,30,45,60,120], labels=["<30","30-45","45-60",">60"])

# Build engineered DF with chosen features + target
feat_list = [
    "SEQN","cycle","AGE","AGE_BIN","SEX","BMI","BMI_CLASS","WAIST_HT_RATIO",
    "SBP_MEAN","DBP_MEAN","HYPERTENSION_FLAG","GLU_VAL","A1C","INSULIN","HOMA_IR",
    "TCHOL","HDL","TRIG","CHOL_HDL_RATIO","TG_TO_HDL","ACR","SMOKER","PA_SCORE","DIABETES_BIN"
]
engineered = df[[c for c in feat_list if c in df.columns]].copy()
print("Engineered shape:", engineered.shape)

# Save cleaned full df and engineered subset
clean_path = OUT_DIR / "nhanes_cleaned_for_diabrisk.csv"
eng_path = OUT_DIR / "nhanes_engineered_for_diabrisk.csv"
df.to_csv(clean_path, index=False)
engineered.to_csv(eng_path, index=False)

summary = {
    "cleaned_path": str(clean_path),
    "engineered_path": str(eng_path),
    "rows": int(df.shape[0]),
    "cols": int(df.shape[1]),
    "engineered_rows": int(engineered.shape[0]),
    "engineered_cols": int(engineered.shape[1]),
    "glucose_source": chosen,
    "glu_non_null": int(df["GLU_VAL"].notna().sum()),
    "a1c_non_null": int(df["A1C"].notna().sum())
}
summary


