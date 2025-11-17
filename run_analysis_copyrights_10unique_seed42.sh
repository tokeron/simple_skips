#!/bin/bash
set -euo pipefail

#############################################################################
# Run analysis on 10 unique copyright characters (one prompt per character)
# Seed 42 only, 1 GPU on interactive partition
# Generates from full specific prompts
#############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCOUNT="${ACCOUNT:-nvr_israel_persdiff}"
PYTHON_PATH="${PYTHON_PATH:-/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python}"
MODEL_NAME="${MODEL_NAME:-black-forest-labs/FLUX.1-dev}"
PROMPTS_CSV="${SCRIPT_DIR}/sample_prompts_with_placeholders_20_3_42.csv"

# Create a filtered CSV with 10 unique copyright characters
FILTERED_CSV="${SCRIPT_DIR}/copyrights_10unique_temp.csv"
${PYTHON_PATH} << 'EOF'
import pandas as pd

df = pd.read_csv('sample_prompts_with_placeholders_20_3_42.csv')
copyrights_df = df[df['subset'] == 'copyrights']

# Get unique copyright items (characters)
unique_items = copyrights_df['original_item'].unique()[:10]

# For each unique character, get first specific prompt
selected_indices = []
for item in unique_items:
    item_rows = copyrights_df[(copyrights_df['original_item'] == item) & pd.notna(copyrights_df['prompt_specific'])]
    if len(item_rows) > 0:
        selected_indices.append(item_rows.iloc[0].name)

# Get selected rows and save
selected_df = df.loc[selected_indices]
selected_df.to_csv('copyrights_10unique_temp.csv', index=False)

print(f"Created filtered CSV with {len(selected_df)} unique characters")
for idx, row in selected_df.iterrows():
    print(f"  Index {idx}: [{row['original_item']}] {row['prompt_specific']}")
EOF

# Analysis configuration
SEEDS="42"
START_IDX=0  # Start from beginning of filtered CSV
END_IDX=10   # All 10 prompts
ANALYSIS_FLAGS="--generate_original_image --do_indirect_effect --do_direct_effect_removing_partial --do_direct_effect_using_only_partial"

# Logging
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_DIR="/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_analysis_copyrights_10unique_s42_${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

echo ""
echo "================================================================================"
echo "Running analysis on 10 unique copyright characters with seed 42"
echo "Logs: ${LOG_DIR}"
echo "Using SLURM account: ${ACCOUNT}"
echo "Seed: ${SEEDS}"
echo "================================================================================"

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
  --job-name="flux_analysis_copyrights_10unique_s42" \
  --output="${LOG_DIR}/analysis.out" \
  --wrap "${CMD}"

echo "Submitted 1-GPU interactive job. Monitor with: tail -f ${LOG_DIR}/analysis.out"


