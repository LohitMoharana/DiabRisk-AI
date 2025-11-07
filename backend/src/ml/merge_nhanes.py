# backend/ml/merge_nhanes.py
import os
import pandas as pd
import pyreadstat

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FILE_PREFIXES = ["DEMO", "BMX", "BPX", "GHB", "GLU", "INS", "TCHOL", "HDL", "TRIGLY", "ALB_CR", "DIQ", "SMQ", "PAQ", "ALQ", "DBQ", "RXQ_RX", "BIOPRO"]

def read_xpt(path):
    try:
        df, meta = pyreadstat.read_xport(path, encoding="latin1")
        return df
    except Exception:
        print(f"Retrying with SAS7BDAT engine fallback: {path}")
        df = pd.read_sas(path, format='xport')
        return df


def find_file_for_prefix(folder, prefix):
    for f in os.listdir(folder):
        if f.upper().startswith(prefix) and f.lower().endswith(".xpt"):
            return os.path.join(folder, f)
    return None

def load_cycle(cycle_folder):
    dfs = []
    for prefix in FILE_PREFIXES:
        fpath = find_file_for_prefix(cycle_folder, prefix)
        if fpath:
            print("Loading", fpath)
            dfs.append(read_xpt(fpath))
        else:
            print("No", prefix, "in", cycle_folder)
    if not dfs:
        return None
    base = dfs[0]
    for d in dfs[1:]:
        base = base.merge(d, on="SEQN", how="outer")
    return base

def main():
    cycles = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    parts = []
    for c in cycles:
        folder = os.path.join(DATA_DIR, c)
        print("Processing cycle:", c)
        df = load_cycle(folder)
        if df is not None:
            df["CYCLE"] = c
            parts.append(df)
    if not parts:
        raise RuntimeError("No cycles loaded. Check DATA_DIR")
    final = pd.concat(parts, ignore_index=True, sort=False)
    out_path = os.path.join(OUTPUT_DIR, "nhanes_raw_merged2.csv")
    final.to_csv(out_path, index=False)
    print("Saved merged raw file to", out_path)
    print("Shape:", final.shape)

if __name__ == "__main__":
    main()
