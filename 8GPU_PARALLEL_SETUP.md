# Full Dataset Image Generation - 8 GPU Parallel Setup

## Overview
Running image generation for 333 prompts across 8 GPUs on the interactive partition (single job, 8 parallel processes).

## Setup Details

### Files Created
1. **`run_single_gpu.py`** - Wrapper that enforces single-GPU mode by:
   - Unsetting all distributed/SLURM environment variables
   - Setting Accelerate to NO distributed mode
   - Running main.py in isolated single-GPU mode

2. **`run_8gpu_parallel.sh`** - Master script that:
   - Launches 8 parallel processes
   - Each process gets one GPU via `CUDA_VISIBLE_DEVICES`
   - Splits 333 prompts across 8 GPUs (~42 prompts each)
   - Logs each GPU's output separately

### Prompt Distribution
- GPU 0: prompts 0-41 (42 prompts)
- GPU 1: prompts 42-83 (42 prompts)
- GPU 2: prompts 84-125 (42 prompts)
- GPU 3: prompts 126-167 (42 prompts)
- GPU 4: prompts 168-209 (42 prompts)
- GPU 5: prompts 210-251 (42 prompts)
- GPU 6: prompts 252-293 (42 prompts)
- GPU 7: prompts 294-332 (39 prompts)

### Output Structure
```
outputs_black-forest-labs/FLUX.1-dev/full_dataset/
├── animals/
│   ├── specific/
│   │   └── <prompt>/
│   │       └── <seed>.jpg
│   ├── placeholder/
│   │   └── <prompt>/
│   │       └── <seed>.jpg
│   └── extension_only/
│       └── <prompt>/
│           └── <seed>.jpg
├── celebrities/
├── copyrights/
├── places/
└── style/
```

## How to Run

### Option 1: Interactive Job (Recommended for Testing)
```bash
# Request an interactive session with 8 GPUs
salloc -p interactive -G 8 --time=4:00:00

# Once in the session, run the script
bash /home/mtoker/git/simple_skips/run_8gpu_parallel.sh
```

### Option 2: Submit as Batch Job
```bash
submit_job -c "bash /home/mtoker/git/simple_skips/run_8gpu_parallel.sh" -g 8 -n flux_full_dataset
```

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### Monitor Logs (Real-time)
```bash
# Find the latest log directory
LOG_DIR=$(ls -td /lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_full_dataset_* | head -1)

# Watch all GPU logs
tail -f ${LOG_DIR}/gpu*.log

# Or watch a specific GPU
tail -f ${LOG_DIR}/gpu0_prompts_0_41.log
```

### Check Progress
```bash
# Count generated images
find /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/ -name '*.jpg' | wc -l

# Expected: 333 prompts × 10 seeds = 3,330 images

# Check by subset
for subset in animals celebrities copyrights places style; do
    count=$(find /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/${subset}/ -name '*.jpg' 2>/dev/null | wc -l)
    echo "${subset}: ${count} images"
done
```

## Key Features

### Skip-if-Exists Logic
- Each prompt/seed combination checks if the output file exists before generation
- Prevents regenerating existing images
- Allows resuming from interruptions

### Single-GPU Enforcement
- Prevents distributed training conflicts
- No EADDRINUSE or DistStoreError issues
- Each process runs independently

### Proper Logging
- Separate log file per GPU
- Timestamped log directories
- Easy to debug individual processes

## Troubleshooting

### If a Process Hangs
```bash
# Find the PID
ps aux | grep run_single_gpu.py

# Kill specific GPU process
kill <PID>
```

### If You Need to Restart
The skip-if-exists logic means you can safely rerun the script - it will only generate missing images.

### Check for Errors
```bash
# Search for errors in all logs
grep -i error ${LOG_DIR}/*.log

# Check for CUDA errors
grep -i cuda ${LOG_DIR}/*.log
```

## Expected Runtime
- Model loading: ~2-5 minutes per GPU
- Per image: ~10-30 seconds (depends on GPU and model)
- Total for 42 images × 10 seeds = 420 images per GPU
- Estimated: 2-4 hours per GPU

## Status
✅ Scripts created and ready to run
✅ Permissions set
✅ Prompt split calculated
🔄 Ready to submit job

## Next Steps
1. Submit the job using one of the options above
2. Monitor logs to ensure all 8 processes start successfully
3. Wait for completion (~2-4 hours)
4. Verify all 3,330 images were generated

