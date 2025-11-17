#!/usr/bin/env python3
"""
Script to reorganize existing images from flat structure to hierarchical structure.

Old structure: sample_prompts/{subset}/{subset}_{variant}_{item}_{extension}_{seed}.jpg
New structure: sample_prompts/{subset}/{variant}/{prompt}/{seed}.jpg
"""

import os
import shutil
import re
from pathlib import Path
import pandas as pd

def parse_filename(filename):
    """
    Parse the old filename format to extract components.
    Format: {subset}_{variant}_{original}_{extension}_{seed}.jpg
    
    Returns: dict with subset, variant, original_item, prompt_extension, seed
    """
    # Remove the .jpg extension
    name = filename.replace('.jpg', '')
    
    # Split by underscore
    parts = name.split('_')
    
    if len(parts) < 5:
        print(f"Warning: Unexpected filename format: {filename}")
        return None
    
    # First part is subset
    subset = parts[0]
    
    # Second part is variant (handle extension_only -> extension_only mapping)
    variant = parts[1]
    
    # Check if this is "extension" followed by "only" (extension_only variant)
    if variant == "extension" and len(parts) > 2 and parts[2] == "only":
        variant = "extension_only"
        # Remove "only" from parts so middle parsing works correctly
        parts = [parts[0], variant] + parts[3:]
    
    # Last part is seed
    seed = parts[-1]
    
    # Everything between variant and seed is original_item and extension
    # We need to figure out where original_item ends and extension begins
    # This is tricky because both can have multiple words
    
    # For now, we'll reconstruct the middle part and match against the CSV
    middle_parts = parts[2:-1]
    
    return {
        'subset': subset,
        'variant': variant,
        'seed': seed,
        'middle': '_'.join(middle_parts),
        'filename': filename
    }

def reorganize_images(base_dir, prompts_csv, dry_run=False):
    """
    Reorganize images from old structure to new structure.
    
    Args:
        base_dir: Base directory containing sample_prompts folder
        prompts_csv: Path to the CSV file with prompt information
        dry_run: If True, only print what would be done without moving files
    """
    # Load the prompts CSV to get the mapping
    df = pd.read_csv(prompts_csv)
    print(f"Loaded {len(df)} prompts from {prompts_csv}")
    
    # Create a mapping from (subset, variant, original_item, extension) to prompt
    prompt_mapping = {}
    for _, row in df.iterrows():
        # Clean the item and extension the same way the old filenames did
        clean_extension = row['prompt_extension'].replace(" ", "_").replace("/", "-")
        clean_item = row['original_item'].replace(" ", "_").replace("/", "-")
        key = f"{clean_item}_{clean_extension}"
        
        prompt_mapping[(row['subset'], row['variant'], key)] = row['prompt']
    
    print(f"Created mapping for {len(prompt_mapping)} unique prompt combinations")
    
    # Walk through the old directory structure
    old_base = Path(base_dir) / "sample_prompts"
    
    if not old_base.exists():
        print(f"Error: Directory not found: {old_base}")
        return
    
    moved_count = 0
    error_count = 0
    
    # Process each subset directory
    for subset_dir in old_base.iterdir():
        if not subset_dir.is_dir():
            continue
            
        subset = subset_dir.name
        print(f"\nProcessing subset: {subset}")
        
        # Get all image files in this subset
        image_files = list(subset_dir.glob("*.jpg"))
        print(f"  Found {len(image_files)} images")
        
        for image_path in image_files:
            # Parse the filename
            parsed = parse_filename(image_path.name)
            
            if parsed is None:
                error_count += 1
                continue
            
            # Look up the prompt from the mapping
            key = (parsed['subset'], parsed['variant'], parsed['middle'])
            
            if key not in prompt_mapping:
                print(f"  Warning: No mapping found for {image_path.name}")
                print(f"    Key: {key}")
                error_count += 1
                continue
            
            prompt = prompt_mapping[key]
            
            # Clean the prompt for directory name
            clean_prompt = prompt.replace(" ", "_").replace("/", "-").replace("\\", "-")
            
            # Create new directory structure
            new_dir = old_base / parsed['subset'] / parsed['variant'] / clean_prompt
            new_path = new_dir / f"{parsed['seed']}.jpg"
            
            # Move the file
            if dry_run:
                print(f"  Would move: {image_path.name}")
                print(f"          to: {new_path.relative_to(old_base)}")
                moved_count += 1
            else:
                # Create directory if it doesn't exist
                new_dir.mkdir(parents=True, exist_ok=True)
                
                # Move the file
                shutil.move(str(image_path), str(new_path))
                moved_count += 1
                
                if moved_count % 100 == 0:
                    print(f"  Moved {moved_count} images...")
    
    print(f"\n{'='*60}")
    if dry_run:
        print("DRY RUN COMPLETE")
        print(f"Would move: {moved_count} images")
    else:
        print("REORGANIZATION COMPLETE")
        print(f"Successfully moved: {moved_count} images")
    print(f"Errors: {error_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Reorganize images into hierarchical structure")
    parser.add_argument('--base_dir', type=str, 
                       default='/home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev',
                       help='Base directory containing sample_prompts folder')
    parser.add_argument('--prompts_csv', type=str,
                       default='/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/sample_prompts_with_placeholders.csv',
                       help='Path to CSV file with prompt information')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print what would be done without actually moving files')
    
    args = parser.parse_args()
    
    reorganize_images(args.base_dir, args.prompts_csv, args.dry_run)

