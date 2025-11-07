import os
import pandas as pd
import psutil
from sklearn.impute import SimpleImputer

RAW_PATH = "../data/processed/nhanes_raw_merged.csv"
OUT_DIR = "../data/processed/"
CLEANED_PATH = os.path.join(OUT_DIR, "nhanes_cleaned.csv")
CHUNK_SIZE = 20000  # adjust if you want faster/slower chunks

def print_mem():
    mem = psutil.virtual_memory()
    print(f"RAM Used: {mem.percent}% | {mem.used/1e9:.2f} GB / {mem.total/1e9:.2f} GB")

def estimate_missing_columns():
    print("\nScanning columns for missing ratio (first pass)...")
    chunks = pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE)
    total_counts = None
    na_counts = None

    for i, chunk in enumerate(chunks):
        print(f" Pass chunk {i+1}")
        total_counts = chunk.notna().sum() if total_counts is None else total_counts + chunk.notna().sum()
        na_counts = chunk.isna().sum() if na_counts is None else na_counts + chunk.isna().sum()

    missing_ratio = na_counts / total_counts
    drop_cols = missing_ratio[missing_ratio > 0.4].index.tolist()
    print(f"Columns >40% missing: {len(drop_cols)}")
    return drop_cols

def clean_data(drop_cols):
    print("\nBeginning main cleaning...")

    # Columns we always keep
    critical = ["LBXGLU", "LBXGH", "BPXSY1"]

    chunks = pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE)
    saved = False

    for i, chunk in enumerate(chunks):
        print(f"\nLoading chunk {i+1}...")
        print_mem()

        # Drop high-missing columns
        chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], errors="ignore")

        # Drop rows missing critical labs
        chunk = chunk.dropna(subset=[c for c in critical if c in chunk.columns])

        # Select numeric columns
        num_cols = chunk.select_dtypes(include=['float', 'int']).columns.tolist()

        # Remove numeric columns that are entirely NaN in this chunk
        valid_num_cols = [c for c in num_cols if chunk[c].notna().sum() > 0]

        dead_cols = set(num_cols) - set(valid_num_cols)
        if dead_cols:
            print(f"Dropping {len(dead_cols)} numeric cols with all NaN in this chunk: {list(dead_cols)[:5]}...")

        # Impute only valid columns
        if valid_num_cols:
            imputer = SimpleImputer(strategy="median")
            chunk[valid_num_cols] = imputer.fit_transform(chunk[valid_num_cols])

        # Save
        if not saved:
            chunk.to_csv(CLEANED_PATH, index=False)
            saved = True
        else:
            chunk.to_csv(CLEANED_PATH, mode="a", header=False, index=False)

        print(f"Chunk {i+1} cleaned and saved.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    drop_cols = estimate_missing_columns()
    clean_data(drop_cols)

    print("\n✅ Finished cleaning!")
    print(f"Cleaned dataset saved at: {CLEANED_PATH}")

if __name__ == "__main__":
    main()
