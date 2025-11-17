#!/usr/bin/env python3
import os
import re
import csv
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path('/home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset')
SUMMARY_CSV = Path('/home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/analysis_summary.csv')
SUMMARY_TXT = Path('/home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/analysis_summary.txt')

# Regex patterns for experiments
PATTERNS = [
    ("indirect", re.compile(r"^(?P<seed>\d+)_src_\d+\.jpg$")),
    ("direct_remove", re.compile(r"^(?P<seed>\d+)_direct_remove_\d+_\d+\.jpg$")),
    ("direct_only", re.compile(r"^(?P<seed>\d+)_direct_only_\d+_\d+\.jpg$")),
    ("direct_and_indirect", re.compile(r"^(?P<seed>\d+)_direct_and_indirect_\d+_\d+\.jpg$")),
]

ORIGINAL_PATTERN = re.compile(r"^(?P<seed>\d+)\.jpg$")

def organize_and_summarize():
    counts = defaultdict(int)  # (subset, prompt_dir, experiment, seed) -> count
    totals_by_experiment = defaultdict(int)
    totals_by_subset = defaultdict(int)

    # Walk subset/prompt directories
    for subset_dir in BASE_DIR.iterdir():
        if not subset_dir.is_dir():
            continue
        specific_dir = subset_dir / 'specific'
        if not specific_dir.is_dir():
            continue
        for prompt_dir in specific_dir.iterdir():
            if not prompt_dir.is_dir():
                continue

            # Experiment subfolders (do not move files; just count)
            exp_dirs = {
                'original': prompt_dir / 'original',
                'indirect': prompt_dir / 'indirect',
                'direct_remove': prompt_dir / 'direct_remove',
                'direct_only': prompt_dir / 'direct_only',
                'direct_and_indirect': prompt_dir / 'direct_and_indirect',
            }
            # Count originals
            if exp_dirs['original'].is_dir():
                for f in exp_dirs['original'].glob('*.jpg'):
                    m0 = ORIGINAL_PATTERN.match(f.name)
                    if not m0:
                        continue
                    seed = int(m0.group('seed'))
                    counts[(subset_dir.name, prompt_dir.name, 'original', seed)] += 1
                    totals_by_experiment['original'] += 1
                    totals_by_subset[subset_dir.name] += 1

            # Count experiment images by regex
            for exp_name, rx in PATTERNS:
                exp_path = exp_dirs.get(exp_name)
                if exp_path and exp_path.is_dir():
                    for f in exp_path.glob('*.jpg'):
                        m = rx.match(f.name)
                        if not m:
                            continue
                        seed = int(m.group('seed'))
                        counts[(subset_dir.name, prompt_dir.name, exp_name, seed)] += 1
                        totals_by_experiment[exp_name] += 1
                        totals_by_subset[subset_dir.name] += 1

    # Write CSV summary
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subset', 'prompt_dir', 'experiment', 'seed', 'count'])
        for (subset, prompt, exp, seed), cnt in sorted(counts.items()):
            w.writerow([subset, prompt, exp, seed, cnt])

    # Write TXT summary
    lines = []
    total_images = sum(totals_by_experiment.values())
    lines.append(f"Total analysis files counted (after organize): {total_images}")
    lines.append("")
    lines.append("By experiment:")
    for exp, v in sorted(totals_by_experiment.items()):
        lines.append(f"  {exp}: {v}")
    lines.append("")
    lines.append("By subset:")
    for subset, v in sorted(totals_by_subset.items()):
        lines.append(f"  {subset}: {v}")
    SUMMARY_TXT.write_text("\n".join(lines))

    print(f"Wrote {SUMMARY_CSV}")
    print(f"Wrote {SUMMARY_TXT}")

if __name__ == '__main__':
    organize_and_summarize()



