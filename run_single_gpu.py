#!/usr/bin/env python3
"""
Wrapper to run main.py or analysis.py with forced single-GPU mode.
This disables all distributed training to avoid conflicts when running multiple processes on one node.
"""

import os
import sys
import subprocess

# Force single-GPU mode - disable all distributed training
# Unset cluster/distributed env vars
for var in [
    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 
    'MASTER_ADDR', 'MASTER_PORT',
    'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS',
    'SLURM_JOB_ID', 'SLURM_JOBID'
]:
    os.environ.pop(var, None)

# Set Accelerate to single process/no distributed
os.environ['ACCELERATE_DISTRIBUTED_TYPE'] = 'NO'
os.environ['ACCELERATE_NUM_PROCESSES'] = '1'
os.environ['ACCELERATE_USE_CPU'] = 'false'
os.environ['ACCELERATE_MIXED_PRECISION'] = 'bf16'

# Run main.py or analysis.py directly as a subprocess with the modified environment
if __name__ == '__main__':
    python_path = '/lustre/fsw/portfolios/nvr/users/mtoker/miniconda3/envs/diffusion_env/bin/python'
    
    # Determine which script to run based on first argument
    if len(sys.argv) > 1 and sys.argv[1] == '--analysis':
        script_path = '/home/mtoker/git/simple_skips/sandbox_diffusers/analysis.py'
        args = sys.argv[2:]  # Remove '--analysis' from args
        print(f"Running analysis.py with args: {args}", flush=True)
    else:
        script_path = '/home/mtoker/git/simple_skips/sandbox_diffusers/main.py'
        args = sys.argv[1:]
        print(f"Running main.py with args: {args}", flush=True)
    
    # Pass through all command line arguments (including --model_name, --device_id, etc.)
    cmd = [python_path, script_path] + args
    
    # Run with the modified environment
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)

