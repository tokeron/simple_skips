#!/bin/bash
set -euo pipefail

#############################################################################
# Run main image generation for seeds 40-60 on polar partition
# Splits 240 prompts across multiple 1-GPU jobs
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${ACCOUNT:-nvr_israel_persdiff}"
PYTHON_PATH="${PYTHON_PATH:-/home/mtoker/.pyenv/versions/3.11.5/bin/python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
PROMPTS_CSV="${SCRIPT_DIR}/sample_prompts_placeholder_extension_only.csv"
SEEDS="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60"
MAIN_GENERATE_VARIANTS="--generate_src_no_trgt --generate_skip_k_layers"

# Job configuration
NUM_JOBS=20  # 20 jobs, each handling 12 prompts
TOTAL_PROMPTS=240

# Logging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_main_seeds40-60_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Running main generation for seeds 40-60 on polar partition at $(date)"
echo "Logs: ${LOG_DIR}"
echo "Using SLURM account: ${ACCOUNT}"
echo "Total prompts: ${TOTAL_PROMPTS}"
echo "Seeds: ${SEEDS}"
echo "Number of jobs: ${NUM_JOBS}"

# Calculate chunk size
CHUNK_SIZE=$((TOTAL_PROMPTS / NUM_JOBS))
echo "Prompts per job: ${CHUNK_SIZE}"

# Submit jobs
for i in $(seq 0 $((NUM_JOBS - 1))); do
  start=$((i * CHUNK_SIZE))
  end=$(((i + 1) * CHUNK_SIZE))
  
  # Last job takes any remainder
  if [ $i -eq $((NUM_JOBS - 1)) ]; then
    end=${TOTAL_PROMPTS}
  fi
  
  job_name="flux_main_s40-60_${i}"
  sbatch_out="${LOG_DIR}/${job_name}.out"
  
  echo "Submitting job ${i}: prompts ${start}-${end}"
  
  sbatch \
    --partition=polar \
    --account=${ACCOUNT} \
    --gres=gpu:1 \
    --cpus-per-task=8 \
    --time=04:00:00 \
    --job-name="${job_name}" \
    --output="${sbatch_out}" \
    --wrap "set -euo pipefail; echo 'Job ${job_name} starting on \$(hostname) at \$(date)'; export PYTHONUNBUFFERED=1; ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py --model_name ${MODEL_NAME} --prompts_csv ${PROMPTS_CSV} ${MAIN_GENERATE_VARIANTS} --seeds ${SEEDS} --start_idx ${start} --end_idx ${end}; echo 'Job ${job_name} completed at \$(date)';" >/dev/null
done

echo "All jobs submitted. Monitor with: squeue -u \$(whoami)"
echo "View logs: tail -f ${LOG_DIR}/*.out"

