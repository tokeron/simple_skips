#!/usr/bin/env python3
"""
Script to create a sample dataset with placeholders.

This script:
1. Loads the prepared prompts
2. For selected subsets, creates placeholder versions
3. Generates a small sample set with specific items, placeholders, and extensions only
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
    'safety': 'a concept',  # Adding placeholder for safety if needed
    'style': 'an artwork'   # Adding placeholder for style if needed
}


def create_sample_dataset(csv_path: str, num_items_per_subset: int = 5, num_endings_per_item: int = 3):
    """
    Create a sample dataset with specific items, placeholders, and extensions.
    
    Args:
        csv_path: Path to the prepared_prompts.csv
        num_items_per_subset: Number of items to select from each subset
        num_endings_per_item: Number of unique extensions to sample per item
    
    Returns:
        DataFrame with sample dataset. Each row contains three prompt variants
        (specific, placeholder, and extension-only) for a selected (item, extension)
        pair.
    """
    # Load the full dataset
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter to only train split for the specified subsets
    target_subsets = ['animals', 'celebrities', 'copyrights', 'places']
    df_train = df[df['split'] == 'train']
    
    # Create sample rows
    sample_rows = []
    
    for subset in target_subsets:
        print(f"\nProcessing {subset}...")
        
        # Get data for this subset
        subset_df = df_train[df_train['subset'] == subset]
        
        if subset_df.empty:
            print(f"  Warning: No data found for {subset}")
            continue
        
        # Get unique items from this subset
        unique_items = subset_df['original_item'].unique()
        
        # Randomly select N items
        selected_items = random.sample(list(unique_items), min(num_items_per_subset, len(unique_items)))
        print(f"  Selected {len(selected_items)} items: {selected_items}")
        
        placeholder = PLACEHOLDERS[subset]
        
        for item_idx, item in enumerate(selected_items):
            # Get all train extensions for this item
            item_data = subset_df[subset_df['original_item'] == item]
            
            if item_data.empty:
                continue
            
            # Get all unique extensions for this item
            extensions = list(item_data['prompt_extension'].unique())

            if not extensions:
                continue

            # Randomly select up to num_endings_per_item extensions (by index for reproducibility)
            selected_extension_indices = random.sample(
                list(range(len(extensions))),
                k=min(num_endings_per_item, len(extensions))
            )

            for selected_extension_idx in selected_extension_indices:
                extension = extensions[selected_extension_idx]

                # Build three prompt variants for a single row
                specific_prompt = f"{item} {extension}"
                placeholder_prompt = f"{placeholder} {extension}"
                extension_only_prompt = extension

                sample_rows.append({
                    'subset': subset,
                    'item_index': item_idx,
                    'extension_index': selected_extension_idx,
                    'original_item': item,
                    'placeholder': placeholder,
                    'prompt_extension': extension,
                    'prompt_specific': specific_prompt,
                    'prompt_placeholder': placeholder_prompt,
                })
    
    # Create DataFrame
    sample_df = pd.DataFrame(sample_rows)
    
    return sample_df


def main(
    num_items_per_subset: int = 10,
    num_endings_per_item: int = 3,
    seed: int = 42,
):
    """Main function."""
    # Set random seed for reproducibility
    random.seed(seed)

    
    # Define paths
    csv_path = "/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/prepared_prompts.csv"
    output_path = "/home/mtoker/git/simple_skips/DiT-Knowledge-Localization/sample_prompts_with_placeholders.csv"
    
    # Create sample dataset
    print("="*80)
    print("CREATING SAMPLE DATASET WITH PLACEHOLDERS")
    print("="*80)
    
    sample_df = create_sample_dataset(csv_path, num_items_per_subset=num_items_per_subset, num_endings_per_item=num_endings_per_item)
    
    # Save to CSV
    sample_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total rows: {len(sample_df)}")
    print(f"\nBy subset:")
    print(sample_df.groupby('subset').size())
    # Each row now contains all three variants; no 'variant' column exists
    
    print(f"\nOutput saved to: {output_path}")
    
    # Show sample rows
    print(f"\n{'='*80}")
    print("SAMPLE PROMPTS")
    print("="*80)
    
    # Show examples from each subset
    # for subset in sample_df['subset'].unique():
    #     print(f"\n{subset.upper()}:")
    #     subset_samples = sample_df[sample_df['subset'] == subset].head(6)
    #     for _, row in subset_samples.iterrows():
    #         print(f"  [item {row['item_index']}, ext_idx {row['extension_index']}]\n"
    #               f"    specific:     {row['prompt_specific']}\n"
    #               f"    placeholder:  {row['prompt_placeholder']}\n"
                #   f"    extension:    {row['prompt_extension']}")
    
    return sample_df


if __name__ == "__main__":
    num_items_per_subset = 20
    num_endings_per_item = 3
    seed = 42
    df = main(num_items_per_subset=num_items_per_subset, num_endings_per_item=num_endings_per_item, seed=seed)
    # save to csv
    save_path = f"sample_prompts_with_placeholders_{num_items_per_subset}_{num_endings_per_item}_{seed}.csv"
    df.to_csv(save_path, index=False)
    print(f"Saved to {save_path}")

