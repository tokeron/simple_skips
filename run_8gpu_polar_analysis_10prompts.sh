#!/bin/bash
#
# Submit 8 single-GPU analysis jobs on the polar partition for
# 10 prompts x 3 seeds x 3 experiments (indirect, direct_remove, direct_only)
#

set -euo pipefail

SCRIPT_DIR="/home/mtoker/git/simple_skips"
PYTHON_PATH="/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python"
LOG_DIR_BASE="/lustre/fsw/portfolios/nvr/users/mtoker/logs"
ACCOUNT=${ACCOUNT:-}

# Configuration
MODEL_NAME=${MODEL_NAME:-black-forest-labs/FLUX.1-dev}
SOURCE_CSV=${SOURCE_CSV:-/home/mtoker/git/simple_skips/sample_prompts_with_placeholders_20_3_42.csv}
TARGET_CSV=${TARGET_CSV:-/home/mtoker/git/simple_skips/sample_prompts_analysis_top10.csv}
NUM_JOBS=${NUM_JOBS:-8}
SEEDS=${SEEDS:-42,52,62}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_LOG_DIR="${LOG_DIR_BASE}/flux_analysis_top10_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

echo "Preparing 10-prompt analysis and submitting ${NUM_JOBS} jobs on polar at $(date)"
echo "Logs: ${RUN_LOG_DIR}"

# Resolve SLURM account if not provided
if [ -z "${ACCOUNT}" ]; then
  ACCOUNT=$(sacctmgr -nP show assoc where user=$(whoami) format=account | head -n1 || true)
fi
if [ -z "${ACCOUNT}" ]; then
  echo "ERROR: No SLURM account found. Set ACCOUNT env var or contact admin." >&2
  exit 1
fi
echo "Using SLURM account: ${ACCOUNT}"

# Create a CSV with the first 10 unique prompt_specific entries (and their rows)
echo "Building 10-prompt CSV: ${TARGET_CSV}"
${PYTHON_PATH} - <<PY
import pandas as pd
src = "${SOURCE_CSV}"
dst = "${TARGET_CSV}"
df = pd.read_csv(src)
# keep first 10 unique prompt_specific rows (stable order)
seen = set()
rows = []
for _, r in df.iterrows():
    ps = r['prompt_specific']
    if ps not in seen:
        seen.add(ps)
        rows.append(r)
    if len(rows) == 10:
        break
out = pd.DataFrame(rows)
out.to_csv(dst, index=False)
print(f"Wrote {dst} with {len(out)} rows")
PY

# Determine total rows in TARGET_CSV (should be 10)
TOTAL=$(${PYTHON_PATH} - <<PY
import pandas as pd
print(len(pd.read_csv('${TARGET_CSV}')))
PY
)

echo "Total prompts (rows): ${TOTAL}"

# Compute splits for ${NUM_JOBS} jobs (ceil division)
CHUNK=$(( (TOTAL + NUM_JOBS - 1) / NUM_JOBS ))
echo "Chunk size per job: ${CHUNK}"

start=0
for job_idx in $(seq 0 $((NUM_JOBS-1))); do
  end=$(( start + CHUNK ))
  if [ ${end} -gt ${TOTAL} ]; then end=${TOTAL}; fi

  if [ ${start} -ge ${end} ]; then
    echo "Skipping job ${job_idx}: no prompts assigned (start=${start}, end=${end})"
    continue
  fi

  job_name="flux_analysis_top10_${job_idx}_${start}_${end}"
  sbatch_out="${RUN_LOG_DIR}/${job_name}.out"

  echo "Submitting job ${job_idx}: prompts ${start}-${end} -> ${sbatch_out}"

  sbatch \
    --partition=polar \
    --account=${ACCOUNT} \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --time=04:00:00 \
    --job-name="${job_name}" \
    --output="${sbatch_out}" \
    --wrap "set -euo pipefail; echo \"Job ${job_name} starting on $(hostname) at $(date)\"; export PYTHONUNBUFFERED=1; ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py --analysis --model_name ${MODEL_NAME} --prompts_csv ${TARGET_CSV} --seeds ${SEEDS} --do_indirect_effect --do_direct_effect_removing_partial --do_direct_effect_using_only_partial --start_idx ${start} --end_idx ${end}; echo \"Job ${job_name} completed at $(date)\";"

  start=${end}
done

echo "All analysis jobs submitted. Monitor with: tail -f ${RUN_LOG_DIR}/*.out"


