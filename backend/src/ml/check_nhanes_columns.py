# backend/src/ml/check_nhanes_columns.py
import pandas as pd
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "src" / "data" / "processed" / "nhanes_raw_merged2.csv"
# change above path if your backup has a different name/location

df = pd.read_csv(PATH, nrows=10)
cols = set(df.columns.str.upper())
interesting = [
    "DIQ010","LBXGH","LBXA1C","LBXGLU","LBXSGL","LBXIN","RXDUSE","RXDDRUG","RXDDRGID",
    "PHAFSTHR","PHAFSTMN","LBXSAL","LBXSCR"
]
print("File:", PATH)
print("Columns present:")
for c in interesting:
    print(f"  {c}: {'YES' if c in cols else 'NO'}")
print("\nAll columns (first 200):")
print(list(df.columns)[:200])
