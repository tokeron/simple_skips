#!/bin/bash
#
# Run 8 parallel processes for analysis.py on sample dataset (20 specific prompts)
# Each process handles indirect/direct effect experiments with 1 random seed per prompt
#

set -e

SCRIPT_DIR="/home/mtoker/git/simple_skips"
PYTHON_PATH="/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python"
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs"
# Optional model/args overrides via environment variables
ANALYSIS_MODEL_NAME=${ANALYSIS_MODEL_NAME:-black-forest-labs/FLUX.1-dev}
ANALYSIS_EXTRA_ARGS=${ANALYSIS_EXTRA_ARGS:-}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_LOG_DIR="${LOG_DIR}/flux_analysis_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

echo "Starting 8-GPU parallel analysis at $(date)"
echo "Log directory: ${RUN_LOG_DIR}"
echo "Experiment: Indirect + Direct effects on sample 'specific' prompts"

# Determine number of rows used by analysis (specific prompts)
TOTAL=$(${PYTHON_PATH} - <<'PY'
import pandas as pd
print(len(pd.read_csv('/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/sample_prompts_with_placeholders.csv')))
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

for gpu_idx in {0..7}; do
    split="${SPLITS[$gpu_idx]}"
    start_idx=$(echo $split | awk '{print $1}')
    end_idx=$(echo $split | awk '{print $2}')
    
    if [ "$start_idx" -ge "$end_idx" ]; then
        echo "Skipping GPU ${gpu_idx}: no prompts assigned"
        continue
    fi
    
    num_prompts=$((end_idx - start_idx))
    log_file="${RUN_LOG_DIR}/gpu${gpu_idx}_prompts_${start_idx}_${end_idx}.log"
    
    echo "Launching GPU ${gpu_idx}: ${num_prompts} prompts (${start_idx}-${end_idx}) -> ${log_file}"
    
    (
        export CUDA_VISIBLE_DEVICES=${gpu_idx}
        ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py \
            --analysis \
            --model_name ${ANALYSIS_MODEL_NAME} \
            --start_idx ${start_idx} \
            --end_idx ${end_idx} \
            ${ANALYSIS_EXTRA_ARGS} \
            > "${log_file}" 2>&1
    ) &
    
    sleep 5
done

echo "All processes launched. Waiting for completion..."
echo "Monitor with: tail -f ${RUN_LOG_DIR}/gpu*.log"

wait

echo "All processes completed at $(date)"
echo "Logs saved to: ${RUN_LOG_DIR}"

# Summary
echo ""
echo "=========================================="
echo "ANALYSIS GENERATION COMPLETE"
echo "=========================================="
echo "Log directory: ${RUN_LOG_DIR}"
echo ""
echo "Check results:"
echo "  ls -la /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/"
echo ""
echo "Count generated images:"
echo "  find /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/ -name '*.jpg' | wc -l"
echo "=========================================="

