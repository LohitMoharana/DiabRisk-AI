"""
upgrade_and_stack.py
Put this file in backend/src/ml/ and run:
python backend/src/ml/upgrade_and_stack.py
"""

import warnings
warnings.filterwarnings("ignore")

import os, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.base import clone
import joblib
import xgboost as xgb

# ========== Config ==========
ROOT = Path(__file__).resolve().parents[2]   # points to backend/
IN_FEATURES = ROOT / "src" / "data" / "processed" / "nhanes_features.csv"
CLEANED_PATH = ROOT / "src" / "data" / "processed" / "nhanes_cleaned.csv"  # used to merge target if absent
OUT_DIR = ROOT / "models" / "upgrade_stack"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ============================

# 1) Load features file
if not IN_FEATURES.exists():
    raise FileNotFoundError(f"{IN_FEATURES} not found — move your nhanes_features.csv there or set IN_FEATURES path.")

df = pd.read_csv(IN_FEATURES)
print("Loaded features:", df.shape)

# 2) Drop direct leakage columns if present
leak_cols = ["A1C","GLUCOSE","INSULIN","HOMA_IR","DIQ_FLAG","DIABETES_BIN","DIABETES_3CLASS","ON_RX"]
df = df.drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore')
print("After dropping leakage cols:", df.shape)

# 3) Ensure TARGET exists; merge from cleaned file if needed
if "DIABETES_BIN" not in df.columns:
    if CLEANED_PATH.exists():
        MASTER_PATH = "../data/processed/nhanes_master_labeled.csv"

        df_orig = pd.read_csv(MASTER_PATH, usecols=["SEQN", "DIABETES_BIN"])

        df = df.merge(df_orig, on="SEQN", how="left")
        print("Merged DIABETES_BIN from cleaned file.")
    else:
        raise RuntimeError("DIABETES_BIN not present in features and cleaned file missing. Provide target.")

df = df[~df["DIABETES_BIN"].isna()].copy()
df["TARGET"] = df["DIABETES_BIN"].astype(int)

# 4) Temporal split (use CYCLE if present)
def get_year_start(c):
    import re
    m = re.search(r'(\d{4})', str(c))
    return int(m.group(1)) if m else np.nan

if "CYCLE" in df.columns:
    df["YEAR_START"] = df["CYCLE"].apply(get_year_start)
    train_df = df[df["YEAR_START"].between(2011,2014)].copy()
    val_df = df[df["YEAR_START"].between(2015,2016)].copy()
    test_df = df[df["YEAR_START"].between(2017,2018)].copy()
else:
    train_df, temp = train_test_split(df, test_size=0.4, stratify=df["TARGET"], random_state=42)
    val_df, test_df = train_test_split(temp, test_size=0.5, stratify=temp["TARGET"], random_state=42)

print("Train/Val/Test shapes:", train_df.shape, val_df.shape, test_df.shape)

# 5) Feature engineering (physiology-based interactions & transforms)
def engineer_features(df_in):
    df = df_in.copy()
    # Interactions
    if "AGE" in df.columns and "BMI" in df.columns:
        df["AGE_x_BMI"] = df["AGE"] * df["BMI"]
    if "BMI" in df.columns and "SBP_MEAN" in df.columns:
        df["BMI_x_SBP"] = df["BMI"] * df["SBP_MEAN"]
    # HDL x smoker
    if "HDL" in df.columns and "SMOKER" in df.columns:
        df["SMOKER_NUM"] = pd.to_numeric(df["SMOKER"], errors="coerce").fillna(0)
        df["HDL_x_SMOKER"] = df["HDL"] * df["SMOKER_NUM"]
    # Age polynomial basis
    if "AGE" in df.columns:
        df["AGE_sq"] = df["AGE"]**2
    # log transforms for skewed vars
    for c in ["ACR","CHOL_HDL_RATIO","TCHOL"]:
        if c in df.columns:
            df[c + "_log"] = np.log1p(df[c].clip(lower=0))
    # waist-to-height bin (quantile)
    if "WAIST_HT_RATIO" in df.columns:
        try:
            df["WHtR_bin"] = pd.qcut(df["WAIST_HT_RATIO"].rank(method="first"), q=5, labels=False, duplicates="drop")
        except Exception:
            df["WHtR_bin"] = pd.cut(df["WAIST_HT_RATIO"], bins=5, labels=False)
    return df

train_df = engineer_features(train_df)
val_df = engineer_features(val_df)
test_df = engineer_features(test_df)

# 6) Build feature list
drop_cols = ["SEQN","CYCLE","YEAR_START","DIABETES_BIN","TARGET"]
feature_cols = [c for c in train_df.columns if c not in drop_cols and train_df[c].nunique() > 1]

numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(train_df[c]) and train_df[c].nunique() > 10]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

print("Selected features:", len(feature_cols), "Numeric:", len(numeric_cols), "Categorical:", len(cat_cols))

# 7) Preprocessing pipelines
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("qt", QuantileTransformer(output_distribution="normal", random_state=0)),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))

])

preproc = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)], remainder="drop", sparse_threshold=0)
preproc.fit(train_df[feature_cols])

X_train = preproc.transform(train_df[feature_cols])
X_val = preproc.transform(val_df[feature_cols])
X_test = preproc.transform(test_df[feature_cols])
y_train = train_df["TARGET"].values
y_val = val_df["TARGET"].values
y_test = test_df["TARGET"].values

joblib.dump(preproc, OUT_DIR / "preprocessor_upgrade.joblib")
print("Preprocessor saved.")

# 8) Base learners
base_log = LogisticRegression(max_iter=2000, class_weight="balanced")
base_xgb = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.05, max_depth=4,
                             n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                             use_label_encoder=False, eval_metric="logloss")

base_log.fit(X_train, y_train)
base_xgb.fit(X_train, y_train)

p_log = base_log.predict_proba(X_test)[:,1]
p_xgb = base_xgb.predict_proba(X_test)[:,1]
print("Base Logistic AUC:", roc_auc_score(y_test, p_log), "AP:", average_precision_score(y_test, p_log))
print("Base XGBoost AUC:", roc_auc_score(y_test, p_xgb), "AP:", average_precision_score(y_test, p_xgb))

# 9) Stacking (OOF preds)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_log = np.zeros(len(train_df)); oof_xgb = np.zeros(len(train_df))

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_df, train_df["TARGET"])):
    X_tr = preproc.transform(train_df.iloc[tr_idx][feature_cols])
    y_tr = train_df.iloc[tr_idx]["TARGET"].values
    X_val_fold = preproc.transform(train_df.iloc[val_idx][feature_cols])

    clf1 = clone(base_log); clf2 = clone(base_xgb)
    clf1.fit(X_tr, y_tr); clf2.fit(X_tr, y_tr)

    oof_log[val_idx] = clf1.predict_proba(X_val_fold)[:,1]
    oof_xgb[val_idx] = clf2.predict_proba(X_val_fold)[:,1]

meta_X = np.vstack([oof_log, oof_xgb]).T
meta_y = train_df["TARGET"].values
meta_clf = LogisticRegression(max_iter=2000)
meta_clf.fit(meta_X, meta_y)

test_stack = np.vstack([p_log, p_xgb]).T
meta_preds = meta_clf.predict_proba(test_stack)[:,1]

print("Stacked AUC:", roc_auc_score(y_test, meta_preds), "AP:", average_precision_score(y_test, meta_preds))
print(classification_report(y_test, (meta_preds>=0.5).astype(int)))

# 10) Persist models & results
joblib.dump(base_log, OUT_DIR / "base_log.joblib")
joblib.dump(base_xgb, OUT_DIR / "base_xgb.joblib")
joblib.dump(meta_clf, OUT_DIR / "meta_clf.joblib")
with open(OUT_DIR / "upgrade_results.json", "w") as f:
    json.dump({
        "base_log": {"auc": float(roc_auc_score(y_test, p_log)), "ap": float(average_precision_score(y_test, p_log))},
        "base_xgb": {"auc": float(roc_auc_score(y_test, p_xgb)), "ap": float(average_precision_score(y_test, p_xgb))},
        "stack_meta": {"auc": float(roc_auc_score(y_test, meta_preds)), "ap": float(average_precision_score(y_test, meta_preds))}
    }, f, indent=2)

print("Saved models and results to", OUT_DIR)
