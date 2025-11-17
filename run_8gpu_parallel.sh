#!/bin/bash
#
# Run 8 parallel image generation processes on 8 GPUs (one job, 8 processes).
# Each process handles a subset of prompts with forced single-GPU mode to avoid distributed conflicts.
#
# Split prompts across 8 GPUs dynamically based on CSV row count

set -e

# Base paths
SCRIPT_DIR="/home/mtoker/git/simple_skips"
PYTHON_PATH="/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs"
# Optional model/args overrides via environment variables
GEN_MODEL_NAME=${GEN_MODEL_NAME:-black-forest-labs/FLUX.1-dev}
GEN_EXTRA_ARGS=${GEN_EXTRA_ARGS:-}

# Create log directory with timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_LOG_DIR="${LOG_DIR}/flux_full_dataset_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

echo "Starting 8-GPU parallel image generation at $(date)"
echo "Log directory: ${RUN_LOG_DIR}"

# Determine number of rows in the CSV used by main.py
# Allow overriding the CSV path via GEN_PROMPTS_CSV
GEN_PROMPTS_CSV=${GEN_PROMPTS_CSV:-/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/sample_prompts_with_placeholders.csv}
TOTAL=$(
  GEN_PROMPTS_CSV="$GEN_PROMPTS_CSV" ${PYTHON_PATH} - <<'PY'
import os, pandas as pd
csv_path = os.environ.get('GEN_PROMPTS_CSV')
print(len(pd.read_csv(csv_path)))
PY
)

if [ -z "$TOTAL" ]; then
    echo "Failed to determine number of prompts; defaulting to 0" >&2
    TOTAL=0
fi

echo "Total prompts (rows): ${TOTAL}"

# Compute splits for 8 GPUs (ceil division)
CHUNK=$(( (TOTAL + 7) / 8 ))
echo "Chunk size per GPU: ${CHUNK}"

declare -a SPLITS
start=0
for gpu_idx in {0..7}; do
    end=$(( start + CHUNK ))
    if [ $end -gt $TOTAL ]; then end=$TOTAL; fi
    SPLITS[$gpu_idx]="$start $end"
    start=$end
done

# Launch 8 processes in parallel, each on its own GPU
for gpu_idx in {0..7}; do
    split="${SPLITS[$gpu_idx]}"
    start_idx=$(echo $split | awk '{print $1}')
    end_idx=$(echo $split | awk '{print $2}')
    
    log_file="${RUN_LOG_DIR}/gpu${gpu_idx}_prompts_${start_idx}_${end_idx}.log"
    
    echo "Launching GPU ${gpu_idx}: prompts ${start_idx}-${end_idx} -> ${log_file}"
    
    # Run in background with CUDA_VISIBLE_DEVICES set
    (
        export CUDA_VISIBLE_DEVICES=${gpu_idx}
        ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py \
            --model_name ${GEN_MODEL_NAME} \
            --prompts_csv ${GEN_PROMPTS_CSV} \
            --generate_variants specific \
            --start_idx ${start_idx} \
            --end_idx ${end_idx} \
            ${GEN_EXTRA_ARGS} \
            > "${log_file}" 2>&1
    ) &
    
    # Small stagger to avoid simultaneous model loading
    sleep 5
done

echo "All 8 processes launched. Waiting for completion..."
echo "Monitor with: tail -f ${RUN_LOG_DIR}/gpu*.log"

# Wait for all background jobs to complete
wait

echo "All processes completed at $(date)"
echo "Logs saved to: ${RUN_LOG_DIR}"

# Summary
echo ""
echo "=========================================="
echo "GENERATION COMPLETE"
echo "=========================================="
echo "Log directory: ${RUN_LOG_DIR}"
echo ""
echo "Check results:"
echo "  ls -la /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/"
echo ""
echo "Count generated images:"
echo "  find /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/ -name '*.jpg' | wc -l"
echo "=========================================="

