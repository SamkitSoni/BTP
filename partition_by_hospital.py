# partition_by_hospital.py
import pandas as pd
import os
import argparse

def partition(inpath="data/diabetic_clean.csv", outdir="clients", min_samples=200, max_clients=None):
    df = pd.read_csv(inpath)
    if "hospitalid" not in df.columns:
        raise ValueError("hospitalid column not found in cleaned CSV.")
    os.makedirs(outdir, exist_ok=True)
    groups = df.groupby("hospitalid")
    valid = [g for _, g in groups if len(g) >= min_samples]
    valid.sort(key=lambda g: len(g), reverse=True)
    if max_clients:
        valid = valid[:max_clients]
    print(f"Creating {len(valid)} client files in {outdir} (min_samples={min_samples})")
    for i, g in enumerate(valid):
        hospital_id = int(g["hospitalid"].iloc[0])
        fname = os.path.join(outdir, f"client_{i}_hospital_{hospital_id}.csv")
        g.drop(columns=["hospitalid"]).to_csv(fname, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inpath", default="data/diabetic_clean.csv")
    parser.add_argument("--out", dest="outdir", default="clients")
    parser.add_argument("--min", dest="min_samples", type=int, default=200)
    parser.add_argument("--max_clients", dest="max_clients", type=int, default=None)
    args = parser.parse_args()
    partition(args.inpath, args.outdir, args.min_samples, args.max_clients)
