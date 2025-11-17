#!/bin/bash
# Script to run 4 parallel jobs, each processing 1/4 of the sample prompts (60 total)

echo "=========================================="
echo "RUNNING 4 PARALLEL JOBS"
echo "Sample dataset: 60 prompts"
echo "Each job: 15 prompts"
echo "=========================================="

# Job 1: Prompts 0-14 (15 prompts)
echo ""
echo "Submitting Job 1: Prompts 0-14"
echo "Modify main.py: start_idx=0, end_idx=15"
echo "Then run:"
echo "submit_job --partition polar4 --nodes 1 --gpu 1 --duration 4.0 --name flux_sample_job1 -c \"/home/mtoker/git/simple_skips/run_main.sh\""
echo ""

# Job 2: Prompts 15-29 (15 prompts)
echo "Submitting Job 2: Prompts 15-29"
echo "Modify main.py: start_idx=15, end_idx=30"
echo "Then run:"
echo "submit_job --partition polar4 --nodes 1 --gpu 1 --duration 4.0 --name flux_sample_job2 -c \"/home/mtoker/git/simple_skips/run_main.sh\""
echo ""

# Job 3: Prompts 30-44 (15 prompts)
echo "Submitting Job 3: Prompts 30-44"
echo "Modify main.py: start_idx=30, end_idx=45"
echo "Then run:"
echo "submit_job --partition polar4 --nodes 1 --gpu 1 --duration 4.0 --name flux_sample_job3 -c \"/home/mtoker/git/simple_skips/run_main.sh\""
echo ""

# Job 4: Prompts 45-59 (15 prompts)
echo "Submitting Job 4: Prompts 45-59"
echo "Modify main.py: start_idx=45, end_idx=None (or 60)"
echo "Then run:"
echo "submit_job --partition polar4 --nodes 1 --gpu 1 --duration 4.0 --name flux_sample_job4 -c \"/home/mtoker/git/simple_skips/run_main.sh\""
echo ""

echo "=========================================="
echo "MANUAL STEPS:"
echo "=========================================="
echo "For each job:"
echo "1. Edit /home/mtoker/git/simple_skips/sandbox_diffusers/main.py"
echo "2. Change start_idx and end_idx on lines ~402-403"
echo "3. Submit the job with the command above"
echo "4. Wait for confirmation before submitting the next"
echo ""
echo "All jobs will write to the same output directory."
echo "The skip mechanism prevents duplicate work."
echo "=========================================="
