#!/usr/bin/env python3
"""
Create a comprehensive dataset CSV with all items from each subset.

For each item in each subset:
- Randomly select one prompt extension from the train set
- Generate 3 variants: specific (item + extension), placeholder (placeholder + extension), extension_only
"""

import pandas as pd
import random
from pathlib import Path

# Define placeholders for each subset
PLACEHOLDERS = {
    'animals': 'an animal',
    'celebrities': 'a person',
    'copyrights': 'a character',
    'places': 'a place',
    'safety': 'a concept',
    'style': 'an artwork'
}

def load_items_and_extensions(subset_path: Path, subset_name: str):
    """Load items and train extensions for a subset."""
    
    # Load items
    if subset_name == "style":
        items_file = subset_path / "wikiart_artists.txt"
    else:
        items_file = subset_path / f"{subset_name}.txt"
    
    if not items_file.exists():
        print(f"Warning: Items file not found: {items_file}")
        return [], []
    
    with open(items_file, 'r') as f:
        items = [line.strip() for line in f if line.strip()]
    
    # Load train extensions
    train_file = subset_path / "train_prompts.txt"
    if not train_file.exists():
        print(f"Warning: Train file not found: {train_file}")
        return items, []
    
    with open(train_file, 'r') as f:
        train_extensions = [line.strip() for line in f if line.strip()]
    
    return items, train_extensions

def create_full_dataset(dataset_dir: str, output_csv: str, subsets: list = None, seed: int = 42):
    """
    Create a comprehensive dataset CSV.
    
    Args:
        dataset_dir: Path to the DiT-Knowledge-Localization directory
        output_csv: Output CSV file path
        subsets: List of subsets to process (None = all)
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    dataset_path = Path(dataset_dir)
    
    if subsets is None:
        # Get all subset directories
        subsets = [d.name for d in dataset_path.iterdir() if d.is_dir()]
    
    all_rows = []
    
    for subset in subsets:
        subset_path = dataset_path / subset
        
        if not subset_path.exists():
            print(f"Warning: Subset directory not found: {subset_path}")
            continue
        
        print(f"\nProcessing subset: {subset}")
        
        # Load items and extensions
        items, train_extensions = load_items_and_extensions(subset_path, subset)
        
        if not items or not train_extensions:
            print(f"  Skipping {subset} - missing items or extensions")
            continue
        
        print(f"  Items: {len(items)}")
        print(f"  Train extensions: {len(train_extensions)}")
        
        # Get placeholder for this subset
        placeholder = PLACEHOLDERS.get(subset, 'something')
        
        # For each item, randomly select one extension
        for item_idx, item in enumerate(items):
            # Randomly select one extension
            extension_idx = random.randint(0, len(train_extensions) - 1)
            extension = train_extensions[extension_idx]
            
            # Special handling for style subset
            if subset == "style":
                # Style: "{extension} in the style of {item}"
                specific_prompt = f"{extension} in the style of {item}"
                placeholder_prompt = f"{extension} in the style of {placeholder}"
                extension_only_prompt = f"{extension}"
            else:
                # Other subsets: "{item} {extension}"
                specific_prompt = f"{item} {extension}"
                placeholder_prompt = f"{placeholder} {extension}"
                extension_only_prompt = extension
            
            # Create 3 variants for this item
            all_rows.extend([
                {
                    'subset': subset,
                    'variant': 'specific',
                    'item_index': item_idx,
                    'extension_index': extension_idx,
                    'original_item': item,
                    'placeholder': None,
                    'prompt_extension': extension,
                    'prompt': specific_prompt
                },
                {
                    'subset': subset,
                    'variant': 'placeholder',
                    'item_index': item_idx,
                    'extension_index': extension_idx,
                    'original_item': item,
                    'placeholder': placeholder,
                    'prompt_extension': extension,
                    'prompt': placeholder_prompt
                },
                {
                    'subset': subset,
                    'variant': 'extension_only',
                    'item_index': item_idx,
                    'extension_index': extension_idx,
                    'original_item': item,
                    'placeholder': None,
                    'prompt_extension': extension,
                    'prompt': extension_only_prompt
                }
            ])
        
        print(f"  Generated {len(items) * 3} prompt variants")
    
    # Create DataFrame and save
    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Dataset creation complete!")
    print(f"Total prompts: {len(df)}")
    print(f"Output: {output_csv}")
    
    if len(df) > 0:
        print(f"\nBreakdown by subset:")
        print(df.groupby('subset').size())
        print(f"\nBreakdown by variant:")
        print(df.groupby('variant').size())
    print(f"{'='*60}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create full dataset CSV for image generation")
    parser.add_argument('--dataset_dir', type=str,
                       default='/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output_csv', type=str,
                       default='/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/full_dataset_prompts.csv',
                       help='Output CSV file path')
    parser.add_argument('--subsets', type=str, nargs='+',
                       default=['animals', 'celebrities', 'copyrights', 'places'],
                       help='List of subsets to process')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    create_full_dataset(args.dataset_dir, args.output_csv, args.subsets, args.seed)

