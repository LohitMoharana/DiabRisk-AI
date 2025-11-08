import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, recall_score
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

print("--- Starting Model Tuning ---")
print("This process may take 10-20 minutes...")

# -------------------------
# 1. Load & Filter Data (Same as pipeline.py)
# -------------------------
try:
    data = pd.read_csv("../data/diabrisk_outputs/nhanes_engineered_for_diabrisk.csv")
except FileNotFoundError:
    # Adjust path if running from root
    try:
        data = pd.read_csv("src/data/diabrisk_outputs/nhanes_engineered_for_diabrisk.csv")
    except FileNotFoundError:
        print("Error: Could not find 'nhanes_engineered_for_diabrisk.csv'.")
        exit()

print(f"Original full dataset shape: {data.shape}")

df_filtered = data[data['GLU_VAL'].notnull() & data['A1C'].notnull()].copy()
print(f"Filtered (fasting subsample) shape: {df_filtered.shape}")

# -------------------------
# 2. Define Features & Target (Same as pipeline.py)
# -------------------------
target_col = 'DIABETES_BIN'
leaky_features = ['GLU_VAL', 'A1C', 'INSULIN', 'HOMA_IR', 'SEQN', 'cycle']
y = df_filtered[target_col]
X = df_filtered.drop(columns=[target_col] + leaky_features, errors='ignore')

# -------------------------
# 3. Preprocessing (Same as pipeline.py)
# -------------------------
# Encode
categorical_cols = X.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Impute
imputer = IterativeImputer(random_state=42, max_iter=10)
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Scale
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)

print("Data preprocessing complete.")

# -------------------------
# 4. Define Parameter Grid for XGBoost
# -------------------------
# Handle class imbalance
scale_pos_weight = (y == 0).sum() / (y == 1).sum()

# Define the model
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale_pos_weight, # Handles imbalance
    n_jobs=-1,
    device="cuda" # Use GPU if available
)

# Define the grid of parameters to search
# We're trying a wide range of values
param_dist = {
    'n_estimators': [100, 250, 500, 750],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5]
}

# -------------------------
# 5. Run Randomized Search
# -------------------------
# We will optimize for 'recall_weighted' to improve our model's
# ability to find the minority class (diabetes).
scoring = 'recall_weighted'

# Set up the search
# n_iter=50 means it will try 50 different combinations
# cv=3 means it will use 3-fold cross-validation
random_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=param_dist,
    n_iter=50,
    scoring = make_scorer(recall_score, pos_label=1),
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

print("Starting randomized search...")
# Run the search on our full, preprocessed data
random_search.fit(X_scaled, y)

# -------------------------
# 6. Show Best Results
# -------------------------
print("\n" + "="*40)
print("--- TUNING COMPLETE ---")
print(f"Best score ({scoring}): {random_search.best_score_:.4f}")
print("Best Parameters Found:")
print(random_search.best_params_)
print("="*40)
print("\nNext steps:")
print("1. Copy these 'Best Parameters' into your 'backend/src/ml/pipeline.py' script.")
print("2. Re-run 'pipeline.py' to generate your new, improved model artifacts.")
print("3. Push the new .pkl files to GitHub.")
