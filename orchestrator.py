# orchestrator.py (updated to auto-load data/features.json)

import subprocess
import time
import glob
import argparse
import os
import json

def load_features(features_arg):
    """
    If --features is provided → use it.
    If not → auto-load from data/features.json.
    """
    if features_arg and len(features_arg) > 0:
        return features_arg

    json_path = "data/features.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "No --features provided and data/features.json not found.\n"
            "Run preprocess.py first."
        )

    with open(json_path, "r") as f:
        feats = json.load(f)
    print(f"[AUTO] Loaded {len(feats)} features from {json_path}")
    return feats

def start_server(rounds=5):
    print("Starting Flower server…")
    return subprocess.Popen(
        ["python", "server.py", "--rounds", str(rounds)]
    )

def start_clients(client_script, client_csvs, features, label="readmit_30",
                  server_addr="127.0.0.1:8080", client_args=None):
    procs = []
    for i, csv_path in enumerate(client_csvs):
        cmd = [
            "python", client_script,
            "--data", csv_path,
            "--client_id", str(i),
            "--label", label,
            "--server-address", server_addr,
        ]

        # auto add feature list
        cmd += ["--features"] + features

        if client_args:
            cmd += client_args

        print("Starting client:", " ".join(cmd))
        p = subprocess.Popen(cmd)
        procs.append(p)
        time.sleep(1.0)

    return procs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_script", default="client_base.py")
    parser.add_argument("--clients_dir", default="clients")
    parser.add_argument("--features", nargs="+", default=None)  # now optional
    parser.add_argument("--label", default="readmit_30")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--client_args", nargs=argparse.REMAINDER, default=None)
    args = parser.parse_args()

    # AUTO LOAD FEATURES
    features_list = load_features(args.features)

    # load client CSVs
    client_files = sorted(glob.glob(os.path.join(args.clients_dir, "*.csv")))
    if not client_files:
        raise SystemExit("No client CSVs found — run partition_by_hospital.py first.")

    # start server
    server_proc = start_server(rounds=args.rounds)
    time.sleep(2)

    # start clients
    client_procs = start_clients(
        args.client_script,
        client_files,
        features_list,
        label=args.label,
        client_args=args.client_args
    )

    try:
        while True:
            alive = any(p.poll() is None for p in client_procs)
            if not alive:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping processes…")
    finally:
        server_proc.terminate()
        for p in client_procs:
            p.terminate()
