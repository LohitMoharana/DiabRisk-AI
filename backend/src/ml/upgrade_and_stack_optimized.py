import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

# ============================================================================
# Load Data
# ============================================================================
df = pd.read_csv("../data/processed/nhanes_raw_merged2.csv")

# Target
y = df["DIABETES_BIN"]
X = df.drop(columns=["DIABETES_BIN"])

# ============================================================================
# Features
# ============================================================================
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# ============================================================================
# Base Models
# ============================================================================
logit = LogisticRegression(
    penalty="l2",
    C=1.2,
    solver="lbfgs",
    max_iter=1000
)

xgb = XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    max_depth=5,
    gamma=0.1,
    eval_metric="logloss",
    tree_method="hist"
)

# Meta model
meta = LogisticRegression(max_iter=500)

# ============================================================================
# Split data BEFORE preprocessing to avoid leakage
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Fit preprocessing on training only
pre.fit(X_train)

X_train_proc = pre.transform(X_train)
X_test_proc = pre.transform(X_test)

# ============================================================================
# OOF Stacking
# ============================================================================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_oof = cross_val_predict(logit, X_train_proc, y_train, cv=kf, method="predict_proba")[:,1]
xgb_oof = cross_val_predict(xgb, X_train_proc, y_train, cv=kf, method="predict_proba")[:,1]

stack_train = np.vstack([log_oof, xgb_oof]).T

# Fit base models fully on processed train
logit.fit(X_train_proc, y_train)
xgb.fit(X_train_proc, y_train)

# Fit meta model
meta.fit(stack_train, y_train)

# ============================================================================
# Test set predictions
# ============================================================================
log_test = logit.predict_proba(X_test_proc)[:,1]
xgb_test = xgb.predict_proba(X_test_proc)[:,1]

stack_test = np.vstack([log_test, xgb_test]).T
stack_pred = meta.predict_proba(stack_test)[:,1]

# Calibrate meta learner for stability
cal = CalibratedClassifierCV(meta, cv=5, method="isotonic")
cal.fit(stack_train, y_train)
stack_pred = cal.predict_proba(stack_test)[:,1]

# ============================================================================
# Metrics
# ============================================================================
auc = roc_auc_score(y_test, stack_pred)
ap = average_precision_score(y_test, stack_pred)

print(f"\nOptimized Stacked AUC: {auc:.4f} AP: {ap:.4f}\n")
print(classification_report(y_test, stack_pred > 0.5))

# ============================================================================
# Save Models
# ============================================================================
joblib.dump(pre, "preprocessor.pkl")
joblib.dump(logit, "logistic.pkl")
joblib.dump(xgb, "xgboost.pkl")
joblib.dump(meta, "meta_model.pkl")
joblib.dump(cal, "calibrated_meta.pkl")

print("✅ Models saved successfully.")
