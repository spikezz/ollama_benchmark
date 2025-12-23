#!/usr/bin/env python3
"""
Generate heatmap from benchmark results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_FILE = "/home/spikezz/Project/ollama_benchmark/benchmark_results.json"
OUTPUT_FILE = "/home/spikezz/Project/ollama_benchmark/heatmap.png"
OUTPUT_CSV = "/home/spikezz/Project/ollama_benchmark/results_table.csv"

def load_results():
    """Load benchmark results"""
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)

def create_heatmap_data(results):
    """Convert results to heatmap format"""
    # Extract unique values
    num_ctx_values = sorted(set(r["num_ctx"] for r in results))
    num_batch_values = sorted(set(r["num_batch"] for r in results))

    # Create matrix
    data = np.full((len(num_batch_values), len(num_ctx_values)), np.nan)

    # Map indices
    ctx_to_idx = {v: i for i, v in enumerate(num_ctx_values)}
    batch_to_idx = {v: i for i, v in enumerate(num_batch_values)}

    # Fill matrix
    for r in results:
        if r.get("prompt_eval_rate") is not None:
            ctx_idx = ctx_to_idx[r["num_ctx"]]
            batch_idx = batch_to_idx[r["num_batch"]]
            data[batch_idx, ctx_idx] = r["prompt_eval_rate"]

    return data, num_ctx_values, num_batch_values

def save_csv(data, num_ctx_values, num_batch_values):
    """Save results as CSV table"""
    import pandas as pd

    df = pd.DataFrame(
        data,
        index=num_batch_values,
        columns=num_ctx_values
    )
    df.index.name = "num_batch"
    df.to_csv(OUTPUT_CSV)
    print(f"CSV table saved to: {OUTPUT_CSV}")

def plot_heatmap(data, num_ctx_values, num_batch_values, metadata):
    """Plot and save heatmap"""
    # Create figure
    plt.figure(figsize=(20, 12))

    # Create heatmap
    ax = sns.heatmap(
        data,
        xticklabels=num_ctx_values,
        yticklabels=num_batch_values,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        cbar_kws={'label': 'Prompt Eval Rate (tokens/s)'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Set labels
    plt.xlabel('num_ctx', fontsize=14, fontweight='bold')
    plt.ylabel('num_batch', fontsize=14, fontweight='bold')
    plt.title('Ollama nemotron_f Model - Prompt Eval Rate Heatmap\n' +
              f'num_ctx range: {num_ctx_values[0]} to {num_ctx_values[-1]}, ' +
              f'num_batch range: {num_batch_values[0]} to {num_batch_values[-1]}',
              fontsize=16, fontweight='bold', pad=20)

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Adjust layout
    plt.tight_layout()

    # Save
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {OUTPUT_FILE}")

    # Also save a version with best value highlighted
    plt.figure(figsize=(20, 12))

    # Find best value
    max_rate = np.nanmax(data)
    max_pos = np.where(data == max_rate)

    ax = sns.heatmap(
        data,
        xticklabels=num_ctx_values,
        yticklabels=num_batch_values,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        cbar_kws={'label': 'Prompt Eval Rate (tokens/s)'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Highlight best value
    if len(max_pos[0]) > 0:
        best_batch_idx = max_pos[0][0]
        best_ctx_idx = max_pos[1][0]
        best_ctx = num_ctx_values[best_ctx_idx]
        best_batch = num_batch_values[best_batch_idx]

        # Add rectangle around best value
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((best_ctx_idx, best_batch_idx), 1, 1,
                               fill=False, edgecolor='blue', lw=4))

        title_text = (f'Ollama nemotron_f Model - Prompt Eval Rate Heatmap\n'
                     f'num_ctx range: {num_ctx_values[0]} to {num_ctx_values[-1]}, '
                     f'num_batch range: {num_batch_values[0]} to {num_batch_values[-1]}\n'
                     f'BEST: num_ctx={best_ctx}, num_batch={best_batch}, rate={max_rate:.2f} tokens/s')
    else:
        title_text = (f'Ollama nemotron_f Model - Prompt Eval Rate Heatmap\n'
                     f'num_ctx range: {num_ctx_values[0]} to {num_ctx_values[-1]}, '
                     f'num_batch range: {num_batch_values[0]} to {num_batch_values[-1]}')

    plt.xlabel('num_ctx', fontsize=14, fontweight='bold')
    plt.ylabel('num_batch', fontsize=14, fontweight='bold')
    plt.title(title_text, fontsize=16, fontweight='bold', pad=20)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    highlighted_output = OUTPUT_FILE.replace('.png', '_highlighted.png')
    plt.savefig(highlighted_output, dpi=300, bbox_inches='tight')
    print(f"Highlighted heatmap saved to: {highlighted_output}")

    return max_rate, best_ctx if len(max_pos[0]) > 0 else None, best_batch if len(max_pos[0]) > 0 else None

def print_statistics(data, num_ctx_values, num_batch_values):
    """Print summary statistics"""
    valid_data = data[~np.isnan(data)]
    total_possible = len(num_ctx_values) * len(num_batch_values)

    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)
    print(f"Completed tests: {len(valid_data)}")
    print(f"Coverage: {len(num_ctx_values)} num_ctx values × {len(num_batch_values)} num_batch values")
    print(f"Possible combinations in current coverage: {total_possible}")
    print(f"Not yet tested (in coverage area): {total_possible - len(valid_data)}")
    print()
    print(f"Max prompt eval rate: {np.nanmax(data):.2f} tokens/s")
    print(f"Min prompt eval rate: {np.nanmin(data):.2f} tokens/s")
    print(f"Mean prompt eval rate: {np.nanmean(data):.2f} tokens/s")
    print(f"Median prompt eval rate: {np.nanmedian(data):.2f} tokens/s")
    print(f"Std deviation: {np.nanstd(data):.2f} tokens/s")
    print("=" * 80)

def main():
    """Main function"""
    print("=" * 80)
    print("Generating Heatmap from Benchmark Results")
    print("=" * 80)
    print()

    # Load results
    print("Loading results...")
    results_data = load_results()
    results = results_data.get("results", [])

    if not results:
        print("ERROR: No results found!")
        return

    print(f"Found {len(results)} test results")
    print()

    # Create heatmap data
    print("Processing data...")
    data, num_ctx_values, num_batch_values = create_heatmap_data(results)

    # Save CSV
    print("Saving CSV table...")
    save_csv(data, num_ctx_values, num_batch_values)
    print()

    # Plot heatmap
    print("Generating heatmap...")
    max_rate, best_ctx, best_batch = plot_heatmap(data, num_ctx_values, num_batch_values, results_data.get("metadata", {}))

    # Print statistics
    print_statistics(data, num_ctx_values, num_batch_values)
    print()

    if best_ctx is not None:
        print("=" * 80)
        print("OPTIMAL CONFIGURATION")
        print("=" * 80)
        print(f"num_ctx: {best_ctx}")
        print(f"num_batch: {best_batch}")
        print(f"Prompt eval rate: {max_rate:.2f} tokens/s")
        print("=" * 80)

    print("\nDone!")

if __name__ == "__main__":
    main()
