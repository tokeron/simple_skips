#!/bin/bash
#
# Submit 6 single-GPU jobs on the polar partition to run MAIN only
# over the sample CSV with 100 seeds. Each job handles a disjoint
# slice of CSV rows.
#

set -euo pipefail

SCRIPT_DIR="/home/mtoker/git/simple_skips"
PYTHON_PATH="/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python"
LOG_DIR_BASE="/lustre/fsw/portfolios/nvr/users/mtoker/logs"
ACCOUNT=${ACCOUNT:-}

# Configuration
MODEL_NAME=${MODEL_NAME:-black-forest-labs/FLUX.1-dev}
PROMPTS_CSV=${PROMPTS_CSV:-/home/mtoker/git/simple_skips/sample_prompts_with_placeholders_20_3_42.csv}
MAIN_GENERATE_VARIANTS=${MAIN_GENERATE_VARIANTS:-specific}
NUM_JOBS=${NUM_JOBS:-6}

# Seeds: generate 0..99
SEEDS=$(python - <<'PY'
print(','.join(str(i) for i in range(100)))
PY
)

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_LOG_DIR="${LOG_DIR_BASE}/flux_dev_polar_main_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

echo "Submitting ${NUM_JOBS} single-GPU MAIN-only jobs to polar at $(date)"
echo "Logs: ${RUN_LOG_DIR}"
echo "Model: ${MODEL_NAME}"
echo "CSV:   ${PROMPTS_CSV}"
echo "MAIN variants: ${MAIN_GENERATE_VARIANTS}"

# Resolve SLURM account if not provided
if [ -z "${ACCOUNT}" ]; then
  ACCOUNT=$(sacctmgr -nP show assoc where user=$(whoami) format=account | head -n1 || true)
fi
if [ -z "${ACCOUNT}" ]; then
  echo "ERROR: No SLURM account found. Set ACCOUNT env var or contact admin." >&2
  exit 1
fi
echo "Using SLURM account: ${ACCOUNT}"

# Determine total rows in CSV
TOTAL=$(${PYTHON_PATH} - <<PY
import pandas as pd
print(len(pd.read_csv('${PROMPTS_CSV}')))
PY
)

if [ -z "${TOTAL}" ]; then
  echo "Failed to determine number of prompts; defaulting to 0" >&2
  TOTAL=0
fi

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

  job_name="fluxdev_polar_main_${job_idx}_${start}_${end}"
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
    --wrap "set -euo pipefail; echo \"Job ${job_name} starting on $(hostname) at $(date)\"; export PYTHONUNBUFFERED=1; ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py --model_name ${MODEL_NAME} --prompts_csv ${PROMPTS_CSV} --generate_variants ${MAIN_GENERATE_VARIANTS} --seeds ${SEEDS} --start_idx ${start} --end_idx ${end}; echo \"Job ${job_name} completed at $(date)\";"

  start=${end}
  sleep 2
done

echo "All MAIN-only jobs submitted. Monitor with: tail -f ${RUN_LOG_DIR}/*.out"


