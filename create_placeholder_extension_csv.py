#!/usr/bin/env python3
"""Create a CSV with only placeholder and extension prompts (no specific)."""
import pandas as pd
import sys

input_csv = sys.argv[1] if len(sys.argv) > 1 else "sample_prompts_with_placeholders_20_3_42.csv"
output_csv = sys.argv[2] if len(sys.argv) > 2 else "sample_prompts_placeholder_extension_only.csv"

df = pd.read_csv(input_csv)
# Exclude 'specific' subset
df_filtered = df[df['subset'] != 'specific'].reset_index(drop=True)

df_filtered.to_csv(output_csv, index=False)
print(f"Filtered {len(df_filtered)} prompts (no specific) from {len(df)} total prompts")
print(f"Wrote: {output_csv}")

