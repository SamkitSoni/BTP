
import glob, pandas as pd, json, sys

with open("data/features.json") as f:
    features = json.load(f)

bad = {}
for path in sorted(glob.glob("clients/client_*_hospital_*.csv")):
    df = pd.read_csv(path, usecols=lambda c: True)
    # report dtype for the features we expect
    non_numeric = []
    for feat in features:
        if feat not in df.columns:
            non_numeric.append((feat, "MISSING"))
        else:
            dtype = df[feat].dtype
            if not (pd.api.types.is_float_dtype(dtype) or pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
                non_numeric.append((feat, str(dtype)))
    if non_numeric:
        bad[path] = non_numeric

if not bad:
    print("All feature columns in all client files look numeric (float/int/bool).")
else:
    print("Found non-numeric features in these client files:\n")
    for p, cols in bad.items():
        print(p)
        for feat, dt in cols:
            print("  ", feat, ":", dt)
        print()
