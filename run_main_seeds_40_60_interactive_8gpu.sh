#!/bin/bash
set -euo pipefail

#############################################################################
# Run main image generation for seeds 40-60 on interactive partition
# Single job with 8 GPUs, processing 240 prompts in parallel
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${ACCOUNT:-nvr_israel_persdiff}"
PYTHON_PATH="${PYTHON_PATH:-/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
PROMPTS_CSV="${SCRIPT_DIR}/sample_prompts_placeholder_extension_only.csv"
SEEDS="40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60"
MAIN_GENERATE_VARIANTS="--generate_src_no_trgt --generate_skip_k_layers"

# Job configuration
NUM_GPUS=8
TOTAL_PROMPTS=240
CHUNK_SIZE=$((TOTAL_PROMPTS / NUM_GPUS))

# Logging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_main_seeds40-60_interactive_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Running main generation for seeds 40-60 on interactive partition at $(date)"
echo "Logs: ${LOG_DIR}"
echo "Using SLURM account: ${ACCOUNT}"
echo "Total prompts: ${TOTAL_PROMPTS}"
echo "Seeds: ${SEEDS}"
echo "GPUs: ${NUM_GPUS}"
echo "Prompts per GPU: ${CHUNK_SIZE}"

# Build the parallel command that will run inside the job
# Each GPU gets a slice of prompts
PARALLEL_CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; "

for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  start=$((gpu_id * CHUNK_SIZE))
  end=$(((gpu_id + 1) * CHUNK_SIZE))
  
  # Last GPU takes any remainder
  if [ $gpu_id -eq $((NUM_GPUS - 1)) ]; then
    end=${TOTAL_PROMPTS}
  fi
  
  gpu_log="${LOG_DIR}/gpu${gpu_id}.log"
  
  PARALLEL_CMD+="(CUDA_VISIBLE_DEVICES=${gpu_id} ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py --model_name ${MODEL_NAME} --prompts_csv ${PROMPTS_CSV} ${MAIN_GENERATE_VARIANTS} --seeds ${SEEDS} --start_idx ${start} --end_idx ${end} > ${gpu_log} 2>&1) & "
done

# Wait for all background processes
PARALLEL_CMD+="wait; echo 'All GPUs finished at \$(date)'"

# Submit single 8-GPU job
sbatch \
  --partition=interactive \
  --account=${ACCOUNT} \
  --gres=gpu:8 \
  --cpus-per-task=64 \
  --time=04:00:00 \
  --job-name="flux_main_s40-60" \
  --output="${LOG_DIR}/main.out" \
  --wrap "${PARALLEL_CMD}"

echo "Submitted single 8-GPU interactive job. Monitor with: tail -f ${LOG_DIR}/*.out ${LOG_DIR}/gpu*.log"

