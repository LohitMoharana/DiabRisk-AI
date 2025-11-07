import pandas as pd
import numpy as np

# ✅ Enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

import xgboost as xgb
import shap
import joblib

# -----------------------------
# 1️⃣ Load engineered dataset
# -----------------------------
df = pd.read_csv("../data/diabrisk_outputs/nhanes_engineered_for_diabrisk.csv")  # your CSV path
print(f"Dataset shape: {df.shape}")
print(df.isnull().sum())

# -----------------------------
# 2️⃣ Separate features and target
# -----------------------------
TARGET = "DIABETES_BIN"
X = df.drop(columns=[TARGET, "SEQN"])
y = df[TARGET]

# -----------------------------
# 3️⃣ Identify categorical and numeric columns
# -----------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
print("Categorical columns:", categorical_cols)

# -----------------------------
# 4️⃣ Preprocessing pipeline
# -----------------------------
# 4a. Iterative Imputer for numeric columns
imputer = IterativeImputer(random_state=42)
X_num_imputed = pd.DataFrame(imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

# 4b. One-hot encode categorical columns
if categorical_cols:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_cat_encoded = pd.DataFrame(
        encoder.fit_transform(X[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols)
    )
    X_preprocessed = pd.concat([X_num_imputed, X_cat_encoded], axis=1)
else:
    X_preprocessed = X_num_imputed

# 4c. Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_preprocessed), columns=X_preprocessed.columns)

# -----------------------------
# 5️⃣ Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# 6️⃣ XGBoost GPU classifier
# -----------------------------
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    tree_method='gpu_hist',       # GPU acceleration
    predictor='gpu_predictor',
    random_state=42,
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    n_iter_no_change=50,          # early stopping
    validation_fraction=0.2,
    verbose=20
)

# -----------------------------
# 7️⃣ Fit the model
# -----------------------------
xgb_clf.fit(X_train, y_train)

# -----------------------------
# 8️⃣ Evaluate
# -----------------------------
y_pred = xgb_clf.predict(X_test)
y_proba = xgb_clf.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC:", roc_auc_score(y_test, y_proba))

# -----------------------------
# 9️⃣ SHAP explanation (PermutationExplainer for GPU-safe)
# -----------------------------
explainer = shap.Explainer(lambda X: xgb_clf.predict_proba(X)[:,1], X_test)
shap_values = explainer(X_test)

# Summary plots
shap.summary_plot(shap_values.values, X_test, plot_type="bar")
shap.summary_plot(shap_values.values, X_test)  # beeswarm

# -----------------------------
# 10️⃣ Save model and preprocessing objects
# -----------------------------
joblib.dump(xgb_clf, "xgb_diabetes_gpu.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")
if categorical_cols:
    joblib.dump(encoder, "encoder.pkl")

print("✅ Pipeline complete. Model and preprocessing saved.")
