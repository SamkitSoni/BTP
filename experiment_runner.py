# experiment_runner.py (updated to auto-load data/features.json)

import os
import shutil
import subprocess
import time
import glob
import pandas as pd
import matplotlib.pyplot as plt
import json

# ---------------------------------------------------------
# Load features automatically from data/features.json
# ---------------------------------------------------------
def load_features():
    json_path = "data/features.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            "ERROR: data/features.json not found.\n"
            "Run preprocess.py first."
        )
    with open(json_path, "r") as f:
        feats = json.load(f)
    print(f"[AUTO] Loaded {len(feats)} features from {json_path}")
    return feats

FEATURES = load_features()

# Experiment configuration
ROUNDS = 5
MAX_CLIENTS = None  # optional: limit #clients (None uses all)
LABEL = "readmit_30"


# ---------------------------------------------------------
# Run orchestrator for an experiment type
# ---------------------------------------------------------
def run_orchestrator(exp_name, client_script, extra_args=None, log_file=None):
    logs_dir = os.path.join("logs", exp_name)

    # Clear previous logs
    if os.path.exists(logs_dir):
        shutil.rmtree(logs_dir)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"\n=== Running {exp_name.upper()} experiment ===")

    # Prepare base command
    # --client_args MUST come last; after this everything goes to the client
    cmd = [
        "python", "orchestrator.py",
        "--client_script", client_script,
        "--clients_dir", "clients",
        "--rounds", str(ROUNDS),
        "--label", LABEL,
        "--client_args",  # all following arguments go to clients
        "--log-dir", logs_dir,
    ]

    # Include DP or ADP hyperparameters
    if extra_args:
        cmd += extra_args

    print("Running:", " ".join(cmd))
    
    # Redirect output to log file if provided
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"=== Running {exp_name.upper()} experiment ===\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'='*80}\n\n")
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run(cmd, check=True)


# ---------------------------------------------------------
# Load logs for experiment
# ---------------------------------------------------------
def load_logs(exp_name):
    folder = os.path.join("logs", exp_name)
    csvs = glob.glob(os.path.join(folder, "client_*.csv"))

    if not csvs:
        print(f"[WARN] No logs found for {exp_name}")
        return None

    dfs = [pd.read_csv(csv).assign(client=os.path.basename(csv)) for csv in csvs]
    df = pd.concat(dfs, ignore_index=True)
    return df


# ---------------------------------------------------------
# Plot results (accuracy curves & epsilon comparison)
# ---------------------------------------------------------
def plot_results(baseline_df, dp_df, adp_df):
    os.makedirs("results", exist_ok=True)

    # 1) Accuracy Over Rounds
    plt.figure(figsize=(8,5))
    if baseline_df is not None:
        b = baseline_df.groupby("round")["val_acc"].mean()
        plt.plot(b.index, b.values, marker="o", label="Baseline")

    if dp_df is not None:
        d = dp_df.groupby("round")["val_acc"].mean()
        plt.plot(d.index, d.values, marker="o", label="DP")

    if adp_df is not None:
        a = adp_df.groupby("round")["val_acc"].mean()
        plt.plot(a.index, a.values, marker="o", label="ADP")

    plt.xlabel("Round")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Over Rounds")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/accuracy_over_rounds.png")
    print("Saved results/accuracy_over_rounds.png")


    # 2) Accuracy vs Epsilon (Privacy-Utility)
    plt.figure(figsize=(8,5))

    if dp_df is not None:
        dp_last = dp_df.sort_values("round").groupby("client").last()
        epsilon = pd.to_numeric(dp_last["epsilon"], errors="coerce")
        plt.scatter(epsilon, dp_last["val_acc"], label="DP", s=60)

    if adp_df is not None:
        adp_last = adp_df.sort_values("round").groupby("client").last()
        epsilon = pd.to_numeric(adp_last["epsilon"], errors="coerce")
        plt.scatter(epsilon, adp_last["val_acc"], label="ADP", s=60)

    if baseline_df is not None:
        b_last = baseline_df.sort_values("round").groupby("client").last()
        b_acc = b_last["val_acc"].mean()
        plt.axhline(b_acc, color="gray", linestyle="--", label="Baseline Accuracy")

    plt.xlabel("Epsilon (per client)")
    plt.ylabel("Final Round Accuracy")
    plt.title("Privacy vs Utility: Accuracy vs Epsilon")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/acc_vs_epsilon.png")
    print("Saved results/acc_vs_epsilon.png")


# ---------------------------------------------------------
# Create summary CSV
# ---------------------------------------------------------
def save_summary(baseline_df, dp_df, adp_df):
    summary_rows = []

    for name, df in [("baseline", baseline_df), ("dp", dp_df), ("adp", adp_df)]:
        if df is None:
            continue
        last = df.sort_values("round").groupby("client").last()
        for idx, row in last.iterrows():
            summary_rows.append({
                "experiment": name,
                "client": idx,
                "final_round": row["round"],
                "val_acc": row["val_acc"],
                "val_loss": row["val_loss"],
                "epsilon": row.get("epsilon", None),
                "noise_multiplier": row.get("noise_multiplier", None),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("results/summary.csv", index=False)
    print("Saved results/summary.csv")


# ---------------------------------------------------------
# MAIN — run all experiments
# ---------------------------------------------------------
def main():
    # Create output log file
    output_log = "results/experiment_output.log"
    os.makedirs("results", exist_ok=True)
    
    # Clear previous log
    with open(output_log, "w") as f:
        f.write("="*80 + "\n")
        f.write("FEDERATED LEARNING EXPERIMENT OUTPUT LOG\n")
        f.write("="*80 + "\n\n")
    
    print("\n[STEP] Running all experiments using features from data/features.json")
    print(f"[INFO] Output will be saved to {output_log}")

    # 1) BASELINE
    run_orchestrator(
        exp_name="baseline",
        client_script="client_base.py",
        extra_args=[],
        log_file=output_log
    )
    baseline_df = load_logs("baseline")

    # 2) DP — fixed noise multiplier (tuned for better performance)
    run_orchestrator(
        exp_name="dp",
        client_script="client_dp.py",
        extra_args=["--noise_multiplier", "0.5", "--local_epochs", "3"],
        log_file=output_log
    )
    dp_df = load_logs("dp")

    # 3) ADP — adaptive noise per round (tuned for better performance)
    # Start with much higher noise (3.0) that decays down to 0.5 over rounds
    run_orchestrator(
        exp_name="adp",
        client_script="client_adp.py",
        extra_args=["--base_noise", "3.0", "--alpha", "0.3", "--min_noise", "0.5", "--local_epochs", "3"],
        log_file=output_log
    )
    adp_df = load_logs("adp")

    # Plot & summarize
    print("\n[STEP] Generating plots and summary...")
    plot_results(baseline_df, dp_df, adp_df)
    save_summary(baseline_df, dp_df, adp_df)

    print("\n=== ALL DONE! ===\n")
    print("Check the results folder:\n"
          " - accuracy_over_rounds.png\n"
          " - acc_vs_epsilon.png\n"
          " - summary.csv\n"
          f" - experiment_output.log\n")

if __name__ == "__main__":
    main()
