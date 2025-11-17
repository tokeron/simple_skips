#!/bin/bash
set -euo pipefail

#############################################################################
# Run analysis on first 5 copyrights prompts (indices 120-124)
# Seed 42 only, 1 GPU on interactive partition
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${ACCOUNT:-nvr_israel_persdiff}"
PYTHON_PATH="${PYTHON_PATH:-/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
PROMPTS_CSV="${SCRIPT_DIR}/sample_prompts_with_placeholders_20_3_42.csv"

# Analysis configuration
SEEDS="42"
START_IDX=120  # First copyrights prompt
END_IDX=125    # First 5 prompts (120-124 inclusive)
ANALYSIS_FLAGS="--generate_original_image --do_indirect_effect --do_direct_effect_removing_partial --do_direct_effect_using_only_partial"

# Logging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_analysis_copyrights_5p_s42_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo "Running analysis on first 5 copyrights prompts (120-124) with seed 42"
echo "Logs: ${LOG_DIR}"
echo "Using SLURM account: ${ACCOUNT}"
echo "Prompts: ${START_IDX} to ${END_IDX}"
echo "Seed: ${SEEDS}"

# Build command
CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; "
CMD+="${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py --analysis "
CMD+="--model_name ${MODEL_NAME} "
CMD+="--prompts_csv ${PROMPTS_CSV} "
CMD+="${ANALYSIS_FLAGS} "
CMD+="--seeds ${SEEDS} "
CMD+="--start_idx ${START_IDX} "
CMD+="--end_idx ${END_IDX}"

# Submit single 1-GPU job
sbatch \
  --partition=interactive \
  --account=${ACCOUNT} \
  --gres=gpu:1 \
  --cpus-per-task=8 \
  --time=04:00:00 \
  --job-name="flux_analysis_copyrights_5p_s42" \
  --output="${LOG_DIR}/analysis.out" \
  --wrap "${CMD}"

echo "Submitted 1-GPU interactive job. Monitor with: tail -f ${LOG_DIR}/analysis.out"


