#!/bin/bash
#
# Submit ONE job on the interactive partition with 8 GPUs.
# Inside the job, launch 8 parallel analysis processes covering
# 10 prompts x 3 seeds x 3 experiments (indirect, direct_remove, direct_only).
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
SEEDS=${SEEDS:-42,52,62}
# Offset/count for selecting unique prompts
PROMPTS_OFFSET=${PROMPTS_OFFSET:-0}
PROMPTS_COUNT=${PROMPTS_COUNT:-10}

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_LOG_DIR="${LOG_DIR_BASE}/flux_analysis_top10_interactive_${TIMESTAMP}"
mkdir -p "${RUN_LOG_DIR}"

echo "Preparing ${PROMPTS_COUNT}-prompt CSV (offset ${PROMPTS_OFFSET}) and submitting a single 8-GPU interactive job at $(date)"
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

# Create a CSV with ${PROMPTS_COUNT} unique prompt_specific entries after skipping ${PROMPTS_OFFSET}
echo "Building ${PROMPTS_COUNT}-prompt CSV (offset ${PROMPTS_OFFSET}): ${TARGET_CSV}"
${PYTHON_PATH} - <<PY
import pandas as pd
src = "${SOURCE_CSV}"
dst = "${TARGET_CSV}"
df = pd.read_csv(src)
seen = set(); uniques = []
for _, r in df.iterrows():
    ps = r['prompt_specific']
    if ps not in seen:
        seen.add(ps); uniques.append(r)
start = int(${PROMPTS_OFFSET})
end = start + int(${PROMPTS_COUNT})
slice_rows = uniques[start:end]
out = pd.DataFrame(slice_rows)
out.to_csv(dst, index=False)
print(f"Wrote {dst} with {len(out)} rows (unique prompts {start}:{end})")
PY

# Determine total rows (should be ${PROMPTS_COUNT})
TOTAL=$(${PYTHON_PATH} - <<PY
import pandas as pd
print(len(pd.read_csv('${TARGET_CSV}')))
PY
)

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

# Build the inner run command that launches 8 processes in parallel
RUN_BLOCK=$(cat <<'INNER'
set -euo pipefail
echo "Job ${SLURM_JOB_NAME} starting on $(hostname) at $(date)"
export PYTHONUNBUFFERED=1
INNER
)

for gpu_idx in {0..7}; do
  split="${SPLITS[$gpu_idx]}"
  s=$(echo $split | awk '{print $1}')
  e=$(echo $split | awk '{print $2}')
  if [ "$s" -ge "$e" ]; then
    continue
  fi
  RUN_BLOCK+=$'\n'
  RUN_BLOCK+=$(cat <<INNER
(
  export CUDA_VISIBLE_DEVICES=${gpu_idx}
  ${PYTHON_PATH} ${SCRIPT_DIR}/run_single_gpu.py \
    --analysis \
    --model_name ${MODEL_NAME} \
    --prompts_csv ${TARGET_CSV} \
    --seeds ${SEEDS} \
    --do_indirect_effect \
    --do_direct_effect_removing_partial \
    --do_direct_effect_using_only_partial \
    --start_idx ${s} \
    --end_idx ${e}
) > ${RUN_LOG_DIR}/gpu${gpu_idx}_prompts_${s}_${e}.log 2>&1 &
sleep 5
INNER
)
done

RUN_BLOCK+=$'\nwait\n'
RUN_BLOCK+=$'echo "All GPU processes completed at $(date)"\n'

sbatch \
  --partition=interactive \
  --account=${ACCOUNT} \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:8 \
  --cpus-per-task=32 \
  --time=04:00:00 \
  --job-name=flux_analysis_top10_interactive \
  --output="${RUN_LOG_DIR}/flux_analysis_top10_interactive.%j.out" \
  --wrap "${RUN_BLOCK}"

echo "Submitted single 8-GPU interactive job. Monitor with: tail -f ${RUN_LOG_DIR}/*.out ${RUN_LOG_DIR}/gpu*.log"


