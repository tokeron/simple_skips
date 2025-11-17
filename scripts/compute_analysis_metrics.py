#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPModel, CLIPProcessor


def clean_prompt_dir_for_analysis(name: str) -> str:
    name = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    return ''.join(c for c in name if (c.isalnum() or c in ('_', '-')))


def load_image(path: Path) -> Image.Image:
    img = Image.open(path).convert('RGB')
    return img


def compute_mse(img_a: Image.Image, img_b: Image.Image) -> float:
    if img_a.size != img_b.size:
        img_b = img_b.resize(img_a.size, Image.BICUBIC)
    a = np.asarray(img_a, dtype=np.float32) / 255.0
    b = np.asarray(img_b, dtype=np.float32) / 255.0
    return float(np.mean((a - b) ** 2))


@torch.no_grad()
def compute_clip_similarity(
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device,
    img_a: Image.Image,
    img_b: Image.Image,
) -> float:
    inputs_a = processor(images=img_a, return_tensors="pt").to(device)
    inputs_b = processor(images=img_b, return_tensors="pt").to(device)
    feat_a = model.get_image_features(**inputs_a)
    feat_b = model.get_image_features(**inputs_b)
    feat_a = torch.nn.functional.normalize(feat_a, dim=-1)
    feat_b = torch.nn.functional.normalize(feat_b, dim=-1)
    sim = (feat_a * feat_b).sum(dim=-1).item()
    return float(sim)


def gather_experiment_files(prompt_dir: Path, experiment: str, seed: int) -> List[Path]:
    exp_dir = prompt_dir / experiment
    if not exp_dir.is_dir():
        return []
    patterns = {
        'indirect': f"{seed}_src_*.jpg",
        'direct_remove': f"{seed}_direct_remove_*.jpg",
        'direct_only': f"{seed}_direct_only_*.jpg",
        'direct_and_indirect': f"{seed}_direct_and_indirect_*.jpg",
    }
    pat = patterns.get(experiment)
    if not pat:
        return []
    return sorted(exp_dir.glob(pat))


def parse_layer_index(experiment: str, file_path: Path) -> Optional[int]:
    name = file_path.name
    # Expected patterns by experiment
    # indirect:  {seed}_src_{L}.jpg
    # direct_remove: {seed}_direct_remove_{L}_{end}.jpg (we use start L)
    # direct_only:   {seed}_direct_only_{L}_{end}.jpg   (we use start L)
    try:
        if experiment == 'indirect':
            # ..._src_{L}.jpg
            base = name.split('_src_')
            if len(base) != 2:
                return None
            tail = base[1]
            layer = int(tail.split('.')[0])
            return layer
        elif experiment in ('direct_remove', 'direct_only'):
            # ..._{exp}_{L}_{end}.jpg
            # split on exp name + '_'
            key = experiment + '_'
            base = name.split(key)
            if len(base) != 2:
                return None
            tail = base[1]
            # tail like: '{L}_{end}.jpg'
            layer = int(tail.split('_')[0])
            return layer
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser(description='Compute MSE and CLIP metrics for analysis outputs')
    parser.add_argument('--base_dir', type=str, default='/home/mtoker/git/simple_skips/outputs_black-forest-labs/FLUX.1-dev/full_dataset',
                        help='Base full_dataset directory')
    parser.add_argument('--prompts_csv', type=str, default='/home/mtoker/git/simple_skips/sample_prompts_analysis_top10.csv',
                        help='Prompts CSV to pick the prompt from')
    parser.add_argument('--prompt_index', type=int, default=0, help='Row index in CSV (default 0)')
    parser.add_argument('--seeds', type=str, default='42,52,62', help='Comma-separated seeds to evaluate')
    parser.add_argument('--experiments', type=str, default='indirect,direct_remove,direct_only',
                        help='Comma-separated experiments to include')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32',
                        help='HF CLIP model id')
    parser.add_argument('--one_minus_mse', action='store_true',
                        help='Plot 1 - MSE instead of raw MSE')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    import pandas as pd
    df = pd.read_csv(args.prompts_csv)
    if not (0 <= args.prompt_index < len(df)):
        raise RuntimeError(f"prompt_index {args.prompt_index} out of range [0, {len(df)-1}]")
    row = df.iloc[args.prompt_index]
    subset = row['subset']
    prompt_specific = str(row['prompt_specific'])
    prompt_dir_name = clean_prompt_dir_for_analysis(prompt_specific)
    prompt_dir = base_dir / subset / 'specific' / prompt_dir_name
    if not prompt_dir.exists():
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    experiments = [e.strip() for e in args.experiments.split(',') if e.strip()]

    # CLIP setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    # Metrics collection
    metrics: Dict[str, List[Tuple[float, float]]] = {e: [] for e in experiments}
    metrics_by_layer: Dict[str, Dict[int, List[Tuple[float, float]]]] = {e: {} for e in experiments}
    details_rows = []  # experiment, seed, layer, file, mse, clip

    for seed in seeds:
        original_path = prompt_dir / 'original' / f'{seed}.jpg'
        if not original_path.exists():
            print(f"Missing original for seed {seed}: {original_path}")
            continue
        orig_img = load_image(original_path)

        for exp in experiments:
            files = gather_experiment_files(prompt_dir, exp, seed)
            if not files:
                print(f"No files for exp={exp} seed={seed}")
                continue
            for f in files:
                try:
                    img = load_image(f)
                    mse = compute_mse(orig_img, img)
                    clip_sim = compute_clip_similarity(clip_model, clip_processor, device, orig_img, img)
                    metrics[exp].append((mse, clip_sim))
                    layer_idx = parse_layer_index(exp, f)
                    if layer_idx is not None:
                        metrics_by_layer.setdefault(exp, {}).setdefault(layer_idx, []).append((mse, clip_sim))
                    details_rows.append((exp, seed, layer_idx if layer_idx is not None else -1, str(f), mse, clip_sim))
                except Exception as e:
                    print(f"Failed on {f}: {e}")

    # Aggregate and plots
    out_metrics_dir = prompt_dir / 'metrics'
    out_metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = base_dir.parent / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    seeds_tag = '-'.join(str(s) for s in seeds)
    exps_tag = '+'.join(experiments)

    # Save per-file CSV
    import csv
    csv_path = out_metrics_dir / 'metrics_per_file.csv'
    with csv_path.open('w', newline='') as fw:
        w = csv.writer(fw)
        w.writerow(['experiment', 'seed', 'layer', 'file', 'mse', 'clip_similarity'])
        for r in details_rows:
            w.writerow(list(r))
    print(f"Wrote {csv_path}")
    # Also write a copy into central figures dir with a meaningful name
    csv_path_global = figures_dir / f"metrics_{subset}_{prompt_dir_name}_seeds-{seeds_tag}_exps-{exps_tag}_per_file.csv"
    try:
        import shutil
        shutil.copyfile(csv_path, csv_path_global)
        print(f"Wrote {csv_path_global}")
    except Exception as e:
        print(f"Failed to copy metrics CSV to figures dir: {e}")

    # Average per experiment
    exp_names = []
    avg_mse_plot = []
    avg_clip = []
    for exp in experiments:
        vals = metrics.get(exp, [])
        if not vals:
            continue
        mses = [v[0] for v in vals]
        clips = [v[1] for v in vals]
        exp_names.append(exp)
        if args.one_minus_mse:
            avg_mse_plot.append(float(np.mean([1.0 - m for m in mses])))
        else:
            avg_mse_plot.append(float(np.mean(mses)))
        avg_clip.append(float(np.mean(clips)))

    # Plot average MSE per experiment
    if exp_names:
        plt.figure(figsize=(6,4))
        plt.bar(exp_names, avg_mse_plot, color='tab:blue')
        plt.ylabel('Average {} vs original'.format('1 - MSE' if args.one_minus_mse else 'MSE'))
        plt.title(('1 - MSE' if args.one_minus_mse else 'MSE') + f' by experiment (subset={subset})')
        plt.tight_layout()
        mse_png = out_metrics_dir / ('avg_mse1m_by_experiment.png' if args.one_minus_mse else 'avg_mse_by_experiment.png')
        plt.savefig(mse_png, dpi=150)
        plt.close()
        print(f"Wrote {mse_png}")
        # Save also to figures dir with context-rich name
        mse_png_global = figures_dir / (f"avg_mse1m_{subset}_{prompt_dir_name}_seeds-{seeds_tag}_exps-{exps_tag}.png" if args.one_minus_mse else f"avg_mse_{subset}_{prompt_dir_name}_seeds-{seeds_tag}_exps-{exps_tag}.png")
        try:
            import shutil
            shutil.copyfile(mse_png, mse_png_global)
            print(f"Wrote {mse_png_global}")
        except Exception as e:
            print(f"Failed to copy MSE plot to figures dir: {e}")

        # Plot average CLIP similarity per experiment
        plt.figure(figsize=(6,4))
        plt.bar(exp_names, avg_clip, color='tab:green')
        plt.ylabel('Average CLIP similarity vs original')
        plt.title(f'CLIP similarity by experiment (subset={subset})')
        plt.tight_layout()
        clip_png = out_metrics_dir / 'avg_clip_by_experiment.png'
        plt.savefig(clip_png, dpi=150)
        plt.close()
        print(f"Wrote {clip_png}")
        # Save also to figures dir with context-rich name
        clip_png_global = figures_dir / f"avg_clip_{subset}_{prompt_dir_name}_seeds-{seeds_tag}_exps-{exps_tag}.png"
        try:
            import shutil
            shutil.copyfile(clip_png, clip_png_global)
            print(f"Wrote {clip_png_global}")
        except Exception as e:
            print(f"Failed to copy CLIP plot to figures dir: {e}")
    else:
        print("No metrics gathered - check that analysis outputs exist.")

    # Layer-wise plots per experiment: mean across seeds
    for exp in experiments:
        layer_map = metrics_by_layer.get(exp, {})
        if not layer_map:
            continue
        layers_sorted = sorted(layer_map.keys())
        if args.one_minus_mse:
            mean_mse_plot = [float(np.mean([1.0 - m for (m, c) in layer_map[L]])) for L in layers_sorted]
        else:
            mean_mse_plot = [float(np.mean([m for (m, c) in layer_map[L]])) for L in layers_sorted]
        mean_clip = [float(np.mean([c for (m, c) in layer_map[L]])) for L in layers_sorted]

        fig, ax1 = plt.subplots(figsize=(7,4))
        ln1 = ax1.plot(layers_sorted, mean_mse_plot, color='tab:blue', marker='o', label=('1 - MSE' if args.one_minus_mse else 'MSE'))
        ax1.set_xlabel('Layer index (start)')
        ax1.set_ylabel(('1 - MSE' if args.one_minus_mse else 'MSE') + ' vs original', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ln2 = ax2.plot(layers_sorted, mean_clip, color='tab:green', marker='s', label='CLIP sim')
        ax2.set_ylabel('CLIP similarity vs original', color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        plt.title(f'{exp} - layer-wise mean (subset={subset}, seeds {seeds_tag})')
        fig.tight_layout()

        layer_png = out_metrics_dir / (f'layerwise_{exp}_mse1m.png' if args.one_minus_mse else f'layerwise_{exp}.png')
        plt.savefig(layer_png, dpi=150)
        plt.close()
        print(f"Wrote {layer_png}")

        # Copy to figures dir with context-rich name
        layer_png_global = figures_dir / (f'layerwise_{exp}_mse1m_{subset}_{prompt_dir_name}_seeds-{seeds_tag}.png' if args.one_minus_mse else f'layerwise_{exp}_{subset}_{prompt_dir_name}_seeds-{seeds_tag}.png')
        try:
            import shutil
            shutil.copyfile(layer_png, layer_png_global)
            print(f"Wrote {layer_png_global}")
        except Exception as e:
            print(f"Failed to copy layerwise plot to figures dir: {e}")


if __name__ == '__main__':
    main()


