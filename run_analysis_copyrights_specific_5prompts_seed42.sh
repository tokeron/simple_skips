#!/bin/bash
set -euo pipefail

#############################################################################
# Run analysis on first 5 copyrights SPECIFIC prompts (indices 120-124)
# Seed 42 only, 1 GPU on interactive partition
# Generates from full specific prompts (not placeholders)
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${ACCOUNT:-nvr_israel_persdiff}"
PYTHON_PATH="${PYTHON_PATH:-/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
PROMPTS_CSV="${SCRIPT_DIR}/sample_prompts_with_placeholders_20_3_42.csv"

# Create a filtered CSV with only the first 5 copyrights specific prompts
FILTERED_CSV="${SCRIPT_DIR}/copyrights_specific_5prompts_temp.csv"
${PYTHON_PATH} << 'EOF'
import pandas as pd
import sys

df = pd.read_csv('sample_prompts_with_placeholders_20_3_42.csv')
# Get copyrights subset with specific prompts
copyrights_specific = df[(df['subset'] == 'copyrights') & pd.notna(df['prompt_specific'])]
# Take first 5
first_5 = copyrights_specific.head(5)
# Save to temp CSV
first_5.to_csv('copyrights_specific_5prompts_temp.csv', index=False)
print(f"Created filtered CSV with {len(first_5)} prompts")
for idx, row in first_5.iterrows():
    print(f"  {idx}: {row['prompt_specific']}")
EOF

# Analysis configuration
SEEDS="42"
START_IDX=0  # Start from beginning of filtered CSV
END_IDX=5    # First 5 prompts
ANALYSIS_FLAGS="--generate_original_image --do_indirect_effect --do_direct_effect_removing_partial --do_direct_effect_using_only_partial"

# Logging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_analysis_copyrights_specific_5p_s42_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo ""
echo "=" * 80
echo "Running analysis on first 5 copyrights SPECIFIC prompts with seed 42"
echo "Logs: ${LOG_DIR}"
echo "Using SLURM account: ${ACCOUNT}"
echo "Seed: ${SEEDS}"
echo "=" * 80

# Build command
CMD="set -euo pipefail; export PYTHONUNBUFFERED=1; "
CMD+="${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py --analysis "
CMD+="--model_name ${MODEL_NAME} "
CMD+="--prompts_csv ${FILTERED_CSV} "
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
  --job-name="flux_analysis_copyrights_specific_5p_s42" \
  --output="${LOG_DIR}/analysis.out" \
  --wrap "${CMD}"

echo "Submitted 1-GPU interactive job. Monitor with: tail -f ${LOG_DIR}/analysis.out"


