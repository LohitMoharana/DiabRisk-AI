# # backend/src/ml/train_compare.py
# import re, os, json, joblib
# import numpy as np, pandas as pd
# from pathlib import Path
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     roc_auc_score, average_precision_score, classification_report,
#     roc_curve, precision_recall_curve
# )
# from xgboost import XGBClassifier
# import matplotlib.pyplot as plt
#
# import torch
# from torch import nn, optim
# from torch.utils.data import Dataset, DataLoader
#
# ROOT = Path(__file__).resolve().parents[2]
# DATA_PATH = ROOT / "src" / "data" / "processed" / "nhanes_features.csv"
# MODELS_DIR = ROOT / "models"
# os.makedirs(MODELS_DIR, exist_ok=True)
#
# df = pd.read_csv(DATA_PATH)
# print("Loaded", df.shape)
#
# def get_year_start(c):
#     m = re.search(r'(\d{4})', str(c))
#     return int(m.group(1)) if m else np.nan
#
# df["YEAR_START"] = df["CYCLE"].apply(get_year_start)
#
# train_df = df[df["YEAR_START"].between(2011, 2014)]
# val_df = df[df["YEAR_START"].between(2015, 2016)]
# test_df = df[df["YEAR_START"].between(2017, 2018)]
#
# print("Train/Val/Test sizes:", train_df.shape, val_df.shape, test_df.shape)
#
# TARGET = "DIABETES_BIN"
# drop_cols = ["SEQN","CYCLE","DIABETES_3CLASS"]
# feature_cols = [c for c in df.columns if c not in drop_cols + [TARGET, "YEAR_START"]]
#
# numeric_cols = [c for c in feature_cols if df[c].dtype in [np.float64, np.int64] and df[c].nunique() > 10]
# cat_cols = [c for c in feature_cols if c not in numeric_cols]
#
# print("Numeric cols:", numeric_cols)
# print("Categorical cols:", cat_cols)
#
# X_train = train_df[feature_cols].copy()
# y_train = train_df[TARGET].astype(int)
# X_val = val_df[feature_cols].copy()
# y_val = val_df[TARGET].astype(int)
# X_test = test_df[feature_cols].copy()
# y_test = test_df[TARGET].astype(int)
#
# numeric_transformer = Pipeline([
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler())
# ])
#
# categorical_transformer = Pipeline([
#     ("imputer", SimpleImputer(strategy="most_frequent")),
#     ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
# ])
#
# preprocessor = ColumnTransformer(
#     [("num", numeric_transformer, numeric_cols),
#      ("cat", categorical_transformer, cat_cols)],
#     remainder="drop",
#     sparse_threshold=0
# )
#
# preprocessor.fit(X_train)
#
# X_train_proc = preprocessor.transform(X_train)
# X_val_proc = preprocessor.transform(X_val)
# X_test_proc = preprocessor.transform(X_test)
# print("Processed shapes:", X_train_proc.shape, X_val_proc.shape, X_test_proc.shape)
#
# joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
#
# # Logistic Regression
# log_pipe = LogisticRegression(max_iter=2000, class_weight="balanced")
# log_pipe.fit(X_train_proc, y_train)
# joblib.dump(log_pipe, MODELS_DIR / "logreg.joblib")
#
# proba_log = log_pipe.predict_proba(X_test_proc)[:,1]
# auc_log = roc_auc_score(y_test, proba_log)
# ap_log = average_precision_score(y_test, proba_log)
# print("Logistic AUC:", auc_log, "AP:", ap_log)
# print(classification_report(y_test, (proba_log>=0.5).astype(int)))
#
# # XGBoost (no early stopping for your version)
# xgb = XGBClassifier(
#     objective="binary:logistic",
#     learning_rate=0.03,
#     max_depth=5,
#     n_estimators=300,
#     subsample=0.85,
#     colsample_bytree=0.85,
#     eval_metric="logloss",
# )
#
# xgb.fit(X_train_proc, y_train)
# joblib.dump(xgb, MODELS_DIR / "xgb.joblib")
#
# proba_xgb = xgb.predict_proba(X_test_proc)[:,1]
# auc_xgb = roc_auc_score(y_test, proba_xgb)
# ap_xgb = average_precision_score(y_test, proba_xgb)
# print("XGB AUC:", auc_xgb, "AP:", ap_xgb)
# print(classification_report(y_test, (proba_xgb>=0.5).astype(int)))
#
# # PyTorch MLP
# class TabularDataset(Dataset):
#     def __init__(self, X, y):
#         self.X = X.astype(np.float32)
#         self.y = y.values.astype(np.float32)
#     def __len__(self): return len(self.y)
#     def __getitem__(self, idx): return self.X[idx], self.y[idx]
#
# Xtr, Xva, Xte = map(lambda x: x.astype(np.float32),
#                     (X_train_proc, X_val_proc, X_test_proc))
# train_ds = TabularDataset(Xtr, y_train)
# val_ds = TabularDataset(Xva, y_val)
# test_ds = TabularDataset(Xte, y_test)
#
# train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=256)
# test_loader = DataLoader(test_ds, batch_size=256)
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
#
# class MLP(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
#             nn.Linear(256,128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
#             nn.Linear(128,1), nn.Sigmoid()
#         )
#     def forward(self,x): return self.net(x).squeeze(-1)
#
# model = MLP(Xtr.shape[1]).to(device)
# opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
# criterion = nn.BCELoss()
#
# best = 0; patience=5; bad=0
# for epoch in range(1,51):
#     model.train(); loss_total=0
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         loss = criterion(model(xb), yb)
#         opt.zero_grad(); loss.backward(); opt.step()
#         loss_total += loss.item()*xb.size(0)
#     avg = loss_total/len(train_loader.dataset)
#
#     model.eval()
#     preds=[]; trues=[]
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             p = model(xb.to(device)).cpu().numpy()
#             preds.append(p); trues.append(yb.numpy())
#     preds=np.concatenate(preds); trues=np.concatenate(trues)
#     try: auc=roc_auc_score(trues,preds)
#     except: auc=0
#
#     print(f"Epoch {epoch} loss={avg:.4f} val_auc={auc:.4f}")
#     if auc>best+1e-4:
#         best=auc; bad=0
#         torch.save(model.state_dict(), MODELS_DIR/"mlp_best.pth")
#     else:
#         bad+=1
#         if bad>=patience:
#             print("Early stop"); break
#
# model.load_state_dict(torch.load(MODELS_DIR/"mlp_best.pth", map_location=device))
# model.eval()
# tp, tt = [], []
# with torch.no_grad():
#     for xb, yb in test_loader:
#         p = model(xb.to(device)).cpu().numpy()
#         tp.append(p); tt.append(yb.numpy())
# tp = np.concatenate(tp); tt = np.concatenate(tt)
#
# auc_mlp = roc_auc_score(tt, tp)
# ap_mlp = average_precision_score(tt, tp)
# print("MLP AUC:", auc_mlp, "AP:", ap_mlp)
# print(classification_report(tt, (tp>=0.5).astype(int)))
#
# results = {
#     "logistic":{"auc":float(auc_log),"ap":float(ap_log)},
#     "xgboost":{"auc":float(auc_xgb),"ap":float(ap_xgb)},
#     "mlp":{"auc":float(auc_mlp),"ap":float(ap_mlp)}
# }
# json.dump(results, open(MODELS_DIR/"results_summary.json","w"), indent=2)
# print(json.dumps(results, indent=2))
#
# # plots
# plt.figure()
# fpr,tpr,_=roc_curve(y_test,proba_xgb); plt.plot(fpr,tpr,label=f"XGB {auc_xgb:.3f}")
# fpr,tpr,_=roc_curve(y_test,proba_log); plt.plot(fpr,tpr,label=f"Log {auc_log:.3f}")
# fpr,tpr,_=roc_curve(tt,tp); plt.plot(fpr,tpr,label=f"MLP {auc_mlp:.3f}")
# plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
# plt.savefig(MODELS_DIR/"roc_comparison.png")
#
# plt.figure()
# prec,rec,_=precision_recall_curve(y_test,proba_xgb); plt.plot(rec,prec,label=f"XGB {ap_xgb:.3f}")
# prec,rec,_=precision_recall_curve(y_test,proba_log); plt.plot(rec,prec,label=f"Log {ap_log:.3f}")
# prec,rec,_=precision_recall_curve(tt,tp); plt.plot(rec,prec,label=f"MLP {ap_mlp:.3f}")
# plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("PR")
# plt.savefig(MODELS_DIR/"pr_comparison.png")
#
# print("All done.")


# backend/src/ml/train_no_leakage.py
import re, os, json, joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "src" / "data" / "processed" / "nhanes_features.csv"
MODELS_DIR = ROOT / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Loaded", df.shape)

# Extract year for temporal split (same as before)
def get_year_start(c):
    m = re.search(r'(\d{4})', str(c))
    return int(m.group(1)) if m else np.nan

df['YEAR_START'] = df['CYCLE'].apply(get_year_start)

# -------------------------
# Drop leakage columns
# -------------------------
leak_cols = [
    "A1C", "GLUCOSE", "INSULIN", "HOMA_IR",
    "DIQ_FLAG", "DIABETES_BIN", "DIABETES_3CLASS",
    "ON_RX", "TRIG"  # optional: TRIG if it's mostly missing or downstream measurement
]
# Also drop any DIQ* or RX* columns you find
leak_cols += [c for c in df.columns if c.upper().startswith("DIQ") or c.upper().startswith("RX")]

leak_cols = list(dict.fromkeys(leak_cols))  # unique
print("Dropping leakage cols (examples):", leak_cols[:10])

# Create a working copy
df_clean = df.copy().drop(columns=[c for c in leak_cols if c in df.columns], errors='ignore')

# -------------------------
# Define target carefully
# For training we need a target. We'll recreate DIABETES_BIN using only self-report OR labs
# but since we've removed DIQ/A1C/GLU, *we need a target*:
#   Option A: reuse original DIABETES_BIN (but that is leakage if DIQ_FLAG used in features)
#   Here we'll assume DIABETES_BIN is available in original file; we will load it from original df
# -------------------------
if "DIABETES_BIN" in df.columns:
    df_clean["TARGET"] = df["DIABETES_BIN"].astype(int)
else:
    raise RuntimeError("DIABETES_BIN target not found in original file. Provide a target column.")

# Temporal split: train 2011-2014, val 2015-2016, test 2017-2018
train_df = df_clean[df["YEAR_START"].between(2011,2014)]
val_df = df_clean[df["YEAR_START"].between(2015,2016)]
test_df = df_clean[df["YEAR_START"].between(2017,2018)]
print("Train/Val/Test sizes:", train_df.shape, val_df.shape, test_df.shape)

# Choose features (drop ID/CYCLE/TARGET/YEAR_START)
drop_cols = ["SEQN","CYCLE","YEAR_START"]
feature_cols = [c for c in df_clean.columns if c not in drop_cols + ["TARGET"]]

# Heuristic: numeric vs categorical
numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_clean[c]) and df_clean[c].nunique() > 10]
cat_cols = [c for c in feature_cols if c not in numeric_cols]

print("Numeric features:", numeric_cols)
print("Categorical features:", cat_cols)

X_train = train_df[feature_cols].copy()
y_train = train_df["TARGET"]
X_val = val_df[feature_cols].copy()
y_val = val_df["TARGET"]
X_test = test_df[feature_cols].copy()
y_test = test_df["TARGET"]

# Preprocessing
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
preprocessor = ColumnTransformer(
    [("num", numeric_transformer, numeric_cols),
     ("cat", categorical_transformer, cat_cols)],
    remainder="drop", sparse_threshold=0
)

preprocessor.fit(X_train)
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

joblib.dump(preprocessor, MODELS_DIR/"preprocessor_no_leakage.joblib")

# Baseline Logistic Regression
log = LogisticRegression(max_iter=2000, class_weight="balanced")
log.fit(X_train_proc, y_train)
joblib.dump(log, MODELS_DIR/"logreg_no_leakage.joblib")
p_log = log.predict_proba(X_test_proc)[:,1]
print("Logistic AUC (no leakage):", roc_auc_score(y_test, p_log), "AP:", average_precision_score(y_test, p_log))
print(classification_report(y_test, (p_log>=0.5).astype(int)))

# XGBoost (legacy-safe)
xgb = XGBClassifier(
    objective="binary:logistic",
    learning_rate=0.03,
    max_depth=5,
    n_estimators=300,
    subsample=0.85,
    colsample_bytree=0.85,
    eval_metric="logloss"
)
xgb.fit(X_train_proc, y_train)
joblib.dump(xgb, MODELS_DIR/"xgb_no_leakage.joblib")
p_x = xgb.predict_proba(X_test_proc)[:,1]
print("XGB AUC (no leakage):", roc_auc_score(y_test, p_x), "AP:", average_precision_score(y_test, p_x))
print(classification_report(y_test, (p_x>=0.5).astype(int)))

# Save results
res = {
    "logistic_no_leak": {"auc": float(roc_auc_score(y_test, p_log)), "ap": float(average_precision_score(y_test, p_log))},
    "xgb_no_leak": {"auc": float(roc_auc_score(y_test, p_x)), "ap": float(average_precision_score(y_test, p_x))}
}
with open(MODELS_DIR/"no_leak_results.json","w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res, indent=2))

# Feature importance (XGBoost)
try:
    importances = xgb.feature_importances_
    # map back feature names (preprocessor transforms)
    # get feature names from preprocessor
    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    ohe_cols = []
    if hasattr(ohe, "get_feature_names_out"):
        ohe_cols = list(ohe.get_feature_names_out(cat_cols))
    else:
        # sklearn older versions
        ohe_cols = []
    feature_names = numeric_cols + ohe_cols
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:20]
    print("Top XGB features (no leak):")
    for name, imp in fi:
        print(f"  {name}: {imp:.4f}")
except Exception as e:
    print("Could not compute feature importances:", e)

print("Saved models in", MODELS_DIR)
