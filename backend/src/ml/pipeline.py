# # Diabetes Risk Prediction: Max-Accuracy XGBoost Pipeline with Early Stopping
# # Author: Lohit Moharana
# # Dataset: Engineered CSV (with labs)
# # Python 3.11+
#
# import pandas as pd
# import numpy as np
# from sklearn.experimental import enable_iterative_imputer  # must enable
# from sklearn.impute import IterativeImputer
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
# import xgboost as xgb
# import joblib
# import shap
# import warnings
# warnings.filterwarnings("ignore")
#
# # -------------------------
# # 1️⃣ Load engineered dataset
# # -------------------------
# data = pd.read_csv("../data/diabrisk_outputs/nhanes_engineered_for_diabrisk.csv")
# print(f"Dataset shape: {data.shape}")
# print(data.isnull().sum())
#
# # -------------------------
# # 2️⃣ Separate features & target
# # -------------------------
# target_col = 'DIABETES_BIN'
# X = data.drop(columns=[target_col])
# y = data[target_col]
#
# # -------------------------
# # 3️⃣ Encode categorical columns
# # -------------------------
# categorical_cols = X.select_dtypes(include=['object']).columns
# print("Categorical columns:", categorical_cols)
#
# if len(categorical_cols) > 0:
#     encoder = OrdinalEncoder()
#     X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
#     print("Categorical columns encoded.")
#
# # -------------------------
# # 4️⃣ Impute missing values
# # -------------------------
# imputer = IterativeImputer(random_state=42)
# X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
# print("Missing values imputed.")
#
# # -------------------------
# # 5️⃣ Scale features
# # -------------------------
# scaler = StandardScaler()
# X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
# print("Features scaled.")
#
# # -------------------------
# # 6️⃣ Train/test split (80/20)
# # -------------------------
# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X_scaled, y, test_size=0.2, random_state=42, stratify=y
# )
#
# # -------------------------
# # 7️⃣ Train/validation split for early stopping (20% of training)
# # -------------------------
# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
# )
#
# # -------------------------
# # 8️⃣ XGBoost model with early stopping
# # -------------------------
# from xgboost import XGBClassifier
#
# xgb_clf = XGBClassifier(
#     objective='binary:logistic',
#     random_state=42,
#     n_estimators=1000,
#     max_depth=5,
#     learning_rate=0.05,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     gamma=0.1,
#     n_iter_no_change=50,          # early stopping
#     validation_fraction=0.2,      # fraction of training data used for early stopping
#     verbose=20
# )
#
# xgb_clf.fit(X_train_full, y_train_full)
#
#
#
# best_model = xgb_clf
#
#
# # -------------------------
# # 9️⃣ Evaluate on test set
# # -------------------------
# y_pred = best_model.predict(X_test)
# y_proba = best_model.predict_proba(X_test)[:, 1]
#
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred))
# print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("\nROC AUC:", roc_auc_score(y_test, y_proba))
#
# # -------------------------
# # 🔟 SHAP feature importance
# # -------------------------
# import shap
#
# # Use TreeExplainer on the underlying booster
# import shap
#
# # Wrap predict_proba for SHAP
# explainer = shap.Explainer(lambda X: best_model.predict_proba(X)[:,1], X_test)
# shap_values = explainer(X_test)
#
# shap.summary_plot(shap_values.values, X_test, plot_type="bar")
#
#
# # -------------------------
# # 11️⃣ Save model & preprocessing objects
# # -------------------------
# joblib.dump(best_model, "diabetes_xgb_model.pkl")
# joblib.dump(imputer, "imputer.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(encoder, "encoder.pkl")
# print("Model, imputer, scaler, and encoder saved successfully!")

# pipeline.py (Corrected for Robust, Leak-Free Model)
# Author: Lohit Moharana
# Dataset: Engineered CSV (with labs)
# Python 3.11+

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import joblib
import shap
import json
import warnings
import os

warnings.filterwarnings("ignore")

# -------------------------
# 1️⃣ Load engineered dataset
# -------------------------
try:
    data = pd.read_csv("../data/diabrisk_outputs/nhanes_engineered_for_diabrisk.csv")
except FileNotFoundError:
    print("Error: The file 'nhanes_engineered_for_diabrisk.csv' was not found.")
    print("Please ensure the path is correct.")
    exit()

print(f"Original full dataset shape: {data.shape}")
print("Original null values (top 15):")
print(data.isnull().sum().sort_values(ascending=False).head(15))

# -------------------------
# 2️⃣ CRITICAL FIX: Define Study Population (Filter to Fasting Subsample)
# -------------------------
# We only build the model on participants who have valid lab data
# for the target variable (GLU_VAL or A1C).
# This drops the ~27,000 rows where these labs were not measured.
#
# !! This is the most important step to prevent data leakage !!
#
df_filtered = data[data['GLU_VAL'].notnull() & data['A1C'].notnull()].copy()

print("-" * 40)
print(f"Filtered (fasting subsample) shape: {df_filtered.shape}")
if df_filtered.shape[0] < 5000:  # Check if filtering was too aggressive
    print("Warning: Filtered dataset is very small. Check your columns.")

# -------------------------
# 3️⃣ Separate features & target (from the *filtered* data)
# -------------------------
target_col = 'DIABETES_BIN'

# Define the features that are "leaky" (i.e., part of the diagnosis)
# These MUST be excluded from the model's features.
leaky_features = ['GLU_VAL', 'A1C', 'INSULIN', 'HOMA_IR', 'SEQN', 'cycle']

# Create X (features) and y (target)
y = df_filtered[target_col]
X = df_filtered.drop(columns=[target_col] + leaky_features, errors='ignore')

# Save the final list of feature columns for the API
model_features = X.columns.tolist()
with open('feature_columns.json', 'w') as f:
    json.dump(model_features, f)

print(f"Target variable defined: '{target_col}'")
print(f"Model will be trained on {len(model_features)} features.")
print("Features:", model_features)

# -------------------------
# 4️⃣ Encode categorical columns
# -------------------------
categorical_cols = X.select_dtypes(include=['object']).columns
print("\nCategorical columns:", categorical_cols)

if len(categorical_cols) > 0:
    # Use handle_unknown='use_encoded_value' and unknown_value=-1 for safety
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])
    print("Categorical columns encoded.")
else:
    encoder = None  # No encoder needed
    print("No categorical columns found to encode.")

# -------------------------
# 5️⃣ Train/test split (80/20)
# -------------------------
# !! MUST split *before* imputation to prevent data leakage !!
X_train_full, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------
# 6️⃣ Train/validation split for early stopping (80/20 of training set)
# -------------------------
X_train, X_val, y_train_val, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# -------------------------
# 7️⃣ Impute missing values (on *non-leaky* features)
# -------------------------
# This now *only* imputes risk factors (e.g., BMI, SBP, TCHOL)
# which is a valid and necessary step.
imputer = IterativeImputer(random_state=42, max_iter=10)

# Fit on the training set
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X.columns)
# Transform validation and test sets
X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X.columns)

print("Missing values imputed (post-split).")

# -------------------------
# 8️⃣ Scale features
# -------------------------
scaler = StandardScaler()

# Fit on the *imputed training set*
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imputed), columns=X.columns)
# Transform validation and test sets
X_val_scaled = pd.DataFrame(scaler.transform(X_val_imputed), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X.columns)

print("Features scaled (post-split).")

# -------------------------
# 9️⃣ XGBoost model with early stopping
# -------------------------
# Handle class imbalance (common in diabetes datasets)
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',  # Use 'logloss' or 'auc' for eval
    random_state=42,
    n_estimators=1000,  # High number, will be stopped early
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    device="cuda",
    scale_pos_weight=scale_pos_weight,  # Handles imbalance
    n_jobs=-1,  # Use all cores
    early_stopping_rounds=50  # <-- ADD THIS LINE HERE  # Use all cores
)

print("\n--- Starting XGBoost Training ---")
xgb_clf.fit(
    X_train_scaled,
    y_train_val,  # Note: y_train was split, use y_train_val
    eval_set=[(X_val_scaled, y_val)],  # Provide the explicit validation set
    verbose=100  # Print progress
)
print("--- Training Complete ---")

# -------------------------
# 10️⃣ Evaluate on test set
# -------------------------
print("\n--- Test Set Evaluation ---")
y_pred = xgb_clf.predict(X_test_scaled)
y_proba = xgb_clf.predict_proba(X_test_scaled)[:, 1]

# These metrics will now be "realistic" (e.g., 85-90% Acc, 0.85-0.90 AUC)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC:", roc_auc_score(y_test, y_proba))

# # -------------------------
# # 11️⃣ SHAP feature importance
# # -------------------------
# print("\nCalculating SHAP values...")
# # Use the optimized TreeExplainer (shap.Explainer auto-selects it)
# # We pass the *model* and *data* (for background distribution)
# explainer = shap.Explainer(lambda x: xgb_clf.predict_proba(x)[:, 1], X_train_scaled)
# shap_values = explainer(X_test_scaled)  # Calculate for test set
#
# # Get the SHAP plot for the "positive" class (Diabetes=1)
# # shap_values.values is [rows, features, classes] for multi-class
# # For binary, it's just [rows, features].
# # If it has 3 dims, use shap_values.values[:,:,1]
# try:
#     if shap_values.values.ndim == 3:
#         shap_values_class1 = shap_values.values[:, :, 1]
#     else:
#         shap_values_class1 = shap_values.values
#
#     # Save the plot
#     shap.summary_plot(shap_values_class1, X_test_scaled, plot_type="bar", show=False)
#     # plt.savefig('shap_summary_plot.png') # Requires matplotlib
#     print("SHAP plot generated (not displayed in console).")
#
# except Exception as e:
#     print(f"Could not generate SHAP plot: {e}")
#
# # -------------------------
# # 12️⃣ Save model & preprocessing objects
# # -------------------------
# print("\nSaving model artifacts...")
# MODEL_DIR = "../models"  # Assuming it's in the backend/src/training folder
# os.makedirs(MODEL_DIR, exist_ok=True)  # Create models dir if it doesn't exist
#
# joblib.dump(xgb_clf, os.path.join(MODEL_DIR, "xgboost_model.pkl"))
# joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
# joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
# if encoder:
#     joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.pkl"))
#
# # Move the feature list to the models directory as well
# os.rename('feature_columns.json', os.path.join(MODEL_DIR, 'feature_columns.json'))
#
# print("Model, imputer, scaler, encoder, and feature list saved successfully!")
# print("Process finished.")