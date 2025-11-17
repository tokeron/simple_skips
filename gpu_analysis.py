import wandb
import pandas as pd

# --- Configuration ---
ENTITY = "nvr-israel"
PROJECT = "dreambooth-flux-dev-lora"

# A list of common metric keys to check for GPU usage
# We only need to find ONE of these to confirm it's a GPU run.
POTENTIAL_GPU_KEYS = [
    "gpu.0.gpu",   # Utilization %
    "gpu.0.mem",   # Memory %
    "gpu.0.temp",  # Temperature
    "gpu.0.util",  # Another common name for utilization
    "cuda.0.gpu",  # CUDA-specific logging
]
# ---------------------

api = wandb.Api()
total_gpu_seconds = 0
gpu_run_count = 0

try:
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    total_runs = len(runs)
    
    print(f"✅ Found {total_runs} runs in project '{ENTITY}/{PROJECT}'.")
    print(f"Analyzing runs for GPU usage (checking {len(POTENTIAL_GPU_KEYS)} common keys)...")

    for i, run in enumerate(runs):
        try:
            # --- THIS IS THE FIX ---
            # We ask for *all* potential keys and check if the resulting
            # DataFrame is empty. We also sample 5 points just in case.
            history_df = run.history(keys=POTENTIAL_GPU_KEYS, samples=5)

            # If the DataFrame is NOT empty, it means at least one
            # of our GPU keys was found in the run's history.
            if not history_df.empty:
                if "_runtime" in run.summary:
                    runtime_seconds = run.summary["_runtime"]
                    total_gpu_seconds += runtime_seconds
                    gpu_run_count += 1
                else:
                    print(f"  - WARNING: Run '{run.name}' (ID: {run.id}) used GPU but has no runtime. Skipping.")
            
            # else:
            #   The DataFrame was empty, so none of the keys were found.
            #   This was a CPU run, and we correctly skip it.

        except Exception as e:
            # This catches errors on a *per-run* basis
            print(f"  - ERROR processing run '{run.name}' (ID: {run.id}): {e}. Skipping this run.")


        # Print progress
        if (i + 1) % 100 == 0 or (i + 1) == total_runs:
            print(f"  ...scanned {i + 1} / {total_runs} runs...")


    total_gpu_hours = total_gpu_seconds / 3600
    total_gpu_days = total_gpu_hours / 24

    print("\n---")
    print("📊 GPU Compute Report (v2)")
    print("---")
    print(f"Total Runs Analyzed: {total_runs}")
    print(f"Runs with GPU Usage: {gpu_run_count}")
    print(f"Total GPU Hours:     {total_gpu_hours:.2f} hours")
    print(f"Total GPU Days:      {total_gpu_days:.2f} days")


except Exception as e:
    print(f"\n❌ An error occurred: {e}")
    print("Please check that your ENTITY and PROJECT names are correct and you are logged in (`wandb login`).")