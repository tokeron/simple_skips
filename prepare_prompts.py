#!/usr/bin/env python3
"""
Script to prepare prompts by combining base items with prompt extensions.

This script:
1. Iterates over subset folders in the dataset directory
2. For each subset, reads the base items and prompt extensions
3. Creates combinations of base items with train/eval prompt extensions
4. Outputs a DataFrame with all combinations
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Tuple


def read_lines(filepath: str) -> List[str]:
    """Read lines from a file, strip whitespace, and filter empty lines."""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()]
        # Filter out empty lines
        lines = [line for line in lines if line]
    return lines


def process_subset(subset_path: Path, subset_name: str) -> pd.DataFrame:
    """
    Process a single subset folder.
    
    Args:
        subset_path: Path to the subset folder
        subset_name: Name of the subset (e.g., "animals")
    
    Returns:
        DataFrame with columns: subset, split, original_item, prompt_extension, full_prompt
    """
    # Define file paths
    # Special case for style: use wikiart_artists.txt instead of style.txt
    if subset_name == "style":
        items_file = subset_path / "wikiart_artists.txt"
    else:
        items_file = subset_path / f"{subset_name}.txt"
    
    train_prompts_file = subset_path / "train_prompts.txt"
    eval_prompts_file = subset_path / "eval_prompts.txt"
    
    # Check if files exist
    if not items_file.exists():
        print(f"Warning: {items_file} not found, skipping {subset_name}")
        return pd.DataFrame()
    
    # Read files
    items = read_lines(str(items_file))
    train_extensions = read_lines(str(train_prompts_file)) if train_prompts_file.exists() else []
    eval_extensions = read_lines(str(eval_prompts_file)) if eval_prompts_file.exists() else []
    
    print(f"Processing {subset_name}:")
    print(f"  - {len(items)} items")
    print(f"  - {len(train_extensions)} train extensions")
    print(f"  - {len(eval_extensions)} eval extensions")
    
    # Create combinations
    rows = []
    
    # Special handling for style subset - prompts are full descriptions, items are artists
    if subset_name == "style":
        # For style: "{prompt} in the style of {artist}"
        for item in items:  # item is an artist name
            for extension in train_extensions:  # extension is a full scene description
                full_prompt = f"{extension} in the style of {item}"
                rows.append({
                    'subset': subset_name,
                    'split': 'train',
                    'original_item': item,
                    'prompt_extension': extension,
                    'full_prompt': full_prompt
                })
        
        for item in items:  # item is an artist name
            for extension in eval_extensions:  # extension is a full scene description
                full_prompt = f"{extension} in the style of {item}"
                rows.append({
                    'subset': subset_name,
                    'split': 'eval',
                    'original_item': item,
                    'prompt_extension': extension,
                    'full_prompt': full_prompt
                })
    else:
        # For other subsets: "{item} {extension}"
        # Process train split
        for item in items:
            for extension in train_extensions:
                full_prompt = f"{item} {extension}"
                rows.append({
                    'subset': subset_name,
                    'split': 'train',
                    'original_item': item,
                    'prompt_extension': extension,
                    'full_prompt': full_prompt
                })
        
        # Process eval split
        for item in items:
            for extension in eval_extensions:
                full_prompt = f"{item} {extension}"
                rows.append({
                    'subset': subset_name,
                    'split': 'eval',
                    'original_item': item,
                    'prompt_extension': extension,
                    'full_prompt': full_prompt
                })
    
    df = pd.DataFrame(rows)
    print(f"  - Generated {len(df)} total prompts ({len(df[df['split']=='train'])} train, {len(df[df['split']=='eval'])} eval)")
    
    return df


def main():
    """Main function to process all subsets."""
    # Define dataset path
    dataset_dir = Path("/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/dataset")
    
    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_dir}")
        return
    
    print(f"Processing dataset from: {dataset_dir}\n")
    
    # Get all subdirectories
    subsets = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if not subsets:
        print("No subsets found in dataset directory")
        return
    
    print(f"Found {len(subsets)} subsets: {[s.name for s in subsets]}\n")
    
    # Process each subset
    all_dfs = []
    for subset_path in sorted(subsets):
        subset_name = subset_path.name
        df = process_subset(subset_path, subset_name)
        if not df.empty:
            all_dfs.append(df)
        print()  # Empty line for readability
    
    # Combine all dataframes
    if not all_dfs:
        print("No data generated")
        return
    
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save to CSV
    output_file = dataset_dir.parent / "prepared_prompts.csv"
    final_df.to_csv(output_file, index=False)
    
    # Print summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total prompts generated: {len(final_df)}")
    print(f"\nBy subset:")
    print(final_df.groupby('subset').size())
    print(f"\nBy split:")
    print(final_df.groupby('split').size())
    print(f"\nBy subset and split:")
    print(final_df.groupby(['subset', 'split']).size())
    print(f"\nOutput saved to: {output_file}")
    
    # Show sample rows
    print(f"\nSample prompts:")
    print(final_df.head(10).to_string(index=False))
    
    return final_df


if __name__ == "__main__":
    df = main()

