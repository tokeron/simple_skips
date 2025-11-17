#!/bin/bash
#
# Monitor the 8-GPU parallel image generation job
#

echo "=========================================="
echo "8-GPU Parallel Job Monitor"
echo "=========================================="
echo ""

# Job ID
JOBID="19083217"
LOG_BASE="/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_full_dataset_20251102-054349"

# Check job status
echo "Job Status:"
squeue -u $USER | grep -E "JOBID|${JOBID}"
echo ""

# Find log directory created by the script
SCRIPT_LOG_DIR=$(find /lustre/fsw/portfolios/nvr/users/mtoker/logs/ -maxdepth 1 -type d -name "flux_full_dataset_*" -newer ${LOG_BASE} 2>/dev/null | head -1)

if [ ! -z "$SCRIPT_LOG_DIR" ]; then
    echo "Script log directory: ${SCRIPT_LOG_DIR}"
    echo ""
    
    # Check each GPU log
    echo "Per-GPU Progress:"
    for gpu in {0..7}; do
        logfile=$(ls ${SCRIPT_LOG_DIR}/gpu${gpu}_*.log 2>/dev/null | head -1)
        if [ -f "$logfile" ]; then
            lines=$(wc -l < "$logfile")
            size=$(du -h "$logfile" | cut -f1)
            last_line=$(tail -1 "$logfile" 2>/dev/null)
            echo "  GPU ${gpu}: ${size} (${lines} lines) - Last: ${last_line}"
        else
            echo "  GPU ${gpu}: No log yet"
        fi
    done
    echo ""
fi

# Check main job logs
echo "Main Job Logs:"
if [ -d "${LOG_BASE}" ]; then
    stdout_log=$(find ${LOG_BASE} -name "*stdout*" -o -name "*.log" | grep -v gpu | head -1)
    if [ -f "$stdout_log" ]; then
        echo "Log file: ${stdout_log}"
        echo "Size: $(du -h ${stdout_log} | cut -f1)"
        echo "Last 5 lines:"
        tail -5 "$stdout_log"
    else
        echo "No stdout log found yet (job may not have started)"
    fi
else
    echo "Log directory not found: ${LOG_BASE}"
fi
echo ""

# Check generated images
echo "Generated Images:"
OUTPUT_DIR="/home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset"
if [ -d "$OUTPUT_DIR" ]; then
    total_images=$(find ${OUTPUT_DIR} -name "*.jpg" 2>/dev/null | wc -l)
    echo "  Total: ${total_images} / 3330 expected"
    
    for subset in animals celebrities copyrights places style; do
        if [ -d "${OUTPUT_DIR}/${subset}" ]; then
            count=$(find ${OUTPUT_DIR}/${subset} -name "*.jpg" 2>/dev/null | wc -l)
            echo "  ${subset}: ${count} images"
        fi
    done
else
    echo "  Output directory not created yet"
fi

echo ""
echo "=========================================="
echo "To view live logs:"
echo "  tail -f ${LOG_BASE}/flux_full_dataset*/commands/*stdout*"
echo "  (or once running, the script creates its own logs)"
echo "=========================================="

