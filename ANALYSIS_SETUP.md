# Analysis Script Setup - Indirect/Direct Effect Experiments

## Overview

The `analysis.py` script runs indirect and direct effect experiments on the sample dataset with the following configuration:

- **Dataset**: 20 "specific" variant prompts from `sample_prompts_with_placeholders.csv`
- **Seeds**: 1 random seed per prompt (assigned deterministically with `random.seed(42)`)
- **Experiments Enabled**:
  - `do_indirect_effect = True` (removes layers from all later layers)
  - `do_direct_effect_removing_partial = True` (removes layers from residual stream)
  - `do_direct_effect_using_only_partial = True` (uses only specific layers)

## Files Created

1. **`/home/mtoker/git/simple_skips/sandbox_diffusers/analysis.py`**
   - Main analysis script
   - Imports `FluxPipeline` and `create_image_grid` from `main.py`
   - Filters to only "specific" variant prompts
   - Assigns one random seed per prompt

2. **`/home/mtoker/git/simple_skips/run_single_gpu.py`** (updated)
   - Now supports `--analysis` flag to run `analysis.py` instead of `main.py`
   - Usage: `python run_single_gpu.py --analysis --start_idx 0 --end_idx 10`

3. **`/home/mtoker/git/simple_skips/run_8gpu_parallel_analysis.sh`**
   - Orchestrates 8 parallel processes for the 20 prompts
   - Splits prompts across GPUs:
     - GPUs 0-2: 3 prompts each
     - GPU 3: 2 prompts
     - GPUs 4-6: 3 prompts each
     - GPU 7: 1 prompt

## Sample Prompts (20 total)

### Animals (5 prompts)
- a cow crying softly
- a zebra crying softly
- a dog as a lego sculpture
- a camel walking along a riverbank
- a tiger under the rain

### Celebrities (5 prompts)
- abraham lincoln sitting on a couch
- tom cruise in a modern apartment
- steve jobs at a train station
- albert einstein sitting on the steps of a building
- taylor swift wearing a leather jacket

### Copyrights (5 prompts)
- the batman basking in the sun
- the king kong yawning sleepily
- the flash made of clay
- the ben 10 running across a grassy field
- the scooby-doo walking through autumn leaves

### Places (5 prompts)
- the burj khalifa with a sunset
- the eiffel tower with a rainbow
- the great wall of china covered in fog
- the sydney opera house in black and white
- the acropolis of athens with falling stars

## Expected Output

### Images per Prompt (with 1 seed)
- **Indirect effect**: ~29 images (source layers 0-56 with step=2)
- **Direct effect removing**: ~5 images (step_size=12)
- **Direct effect using only**: ~5 images (step_size=12)
- **Total per prompt**: ~39 images
- **Grand total**: 20 prompts × 39 images = **~780 images**

### Output Directory Structure
```
outputs_black-forest-labs/FLUX.1-dev/full_dataset/
├── animals/
│   └── specific/
│       ├── a_cow_crying_softly/
│       │   ├── 80.jpg (if generate_original_image=True)
│       │   ├── 80_src_0.jpg
│       │   ├── 80_src_2.jpg
│       │   ├── ...
│       │   ├── 80_direct_remove_0_12.jpg
│       │   ├── 80_direct_remove_12_24.jpg
│       │   ├── ...
│       │   ├── 80_direct_only_0_12.jpg
│       │   ├── 80_direct_only_12_24.jpg
│       │   └── ...
│       └── ...
├── celebrities/
│   └── specific/
│       └── ...
├── copyrights/
│   └── specific/
│       └── ...
└── places/
    └── specific/
        └── ...
```

## Running the Analysis

### Submit to Cluster (Recommended)
```bash
cd /home/mtoker/git/simple_skips
submit_job --partition interactive --gpu 8 --nodes 1 --duration 4.0 --name flux_analysis \
  -c "bash /home/mtoker/git/simple_skips/run_8gpu_parallel_analysis.sh"
```

### Run Locally (Single GPU)
```bash
cd /home/mtoker/git/simple_skips
python run_single_gpu.py --analysis --start_idx 0 --end_idx 5
```

## Monitoring

### Check Job Status
```bash
squeue -u $USER | grep flux
```

### Watch Logs
```bash
# Find latest log directory
LOG_DIR=$(ls -td /lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_analysis_* | head -1)

# Watch all GPUs
tail -f ${LOG_DIR}/gpu*.log

# Watch specific GPU
tail -f ${LOG_DIR}/gpu0_prompts_0_3.log
```

### Count Generated Images
```bash
find /home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset/ -name '*.jpg' | wc -l
```

### Check WandB
- Project: `direct_effect_exp`
- URL: https://wandb.ai/nvr-israel/direct_effect_exp

## Current Status

**Job ID**: 19334965  
**Status**: Running  
**Node**: batch-block1-10019  
**Log Directory**: `/lustre/fsw/portfolios/nvr/users/mtoker/logs/flux_analysis_20251103-131251/`

All 8 GPUs are currently loading models and will begin generating images shortly.

## Estimated Time

- **Model loading**: ~7 minutes per GPU (already in progress)
- **Generation**: ~30-60 minutes for all 780 images across 8 GPUs
- **Total**: ~40-70 minutes

## Notes

- Each prompt gets a deterministically assigned random seed (using `random.seed(42)`)
- Seeds range from 40-99
- File existence checks prevent regeneration of existing images
- All experiments log to WandB with detailed metrics
- Grids are automatically created for visualization



