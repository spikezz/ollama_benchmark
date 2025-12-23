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
OUTPUT_FILE = "/home/spikezz/Project/ollama_benchmark/heatmap_highlighted.png"
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

    # Create matrix for data and errors
    data = np.full((len(num_batch_values), len(num_ctx_values)), np.nan)
    errors = {}  # Store error information: (batch_idx, ctx_idx) -> error_type

    # Map indices
    ctx_to_idx = {v: i for i, v in enumerate(num_ctx_values)}
    batch_to_idx = {v: i for i, v in enumerate(num_batch_values)}

    # Fill matrix
    for r in results:
        ctx_idx = ctx_to_idx[r["num_ctx"]]
        batch_idx = batch_to_idx[r["num_batch"]]

        if r.get("prompt_eval_rate") is not None:
            data[batch_idx, ctx_idx] = r["prompt_eval_rate"]
        elif r.get("error"):
            # Mark as special value for failed tests (use -1 as marker)
            data[batch_idx, ctx_idx] = -1
            errors[(batch_idx, ctx_idx)] = r["error"]

    return data, num_ctx_values, num_batch_values, errors

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

def plot_heatmap(data, num_ctx_values, num_batch_values, metadata, errors):
    """Plot and save heatmap with best value highlighted"""
    # Create figure with dynamic size based on number of values
    # Increase width to avoid overlapping annotations
    fig_width = max(20, len(num_ctx_values) * 0.8)
    fig_height = max(6, len(num_batch_values) * 0.4)

    # Mask for failed tests
    failed_mask = (data == -1)

    # Replace -1 with nan for display purposes (will show as gray)
    display_data = data.copy()
    display_data[failed_mask] = np.nan

    plt.figure(figsize=(fig_width, fig_height))

    # Find best value (excluding failed tests)
    max_rate = np.nanmax(display_data)
    max_pos = np.where(display_data == max_rate)

    ax = sns.heatmap(
        display_data,
        xticklabels=num_ctx_values,
        yticklabels=num_batch_values,
        cmap="YlOrRd",
        annot=True,
        fmt=".1f",
        annot_kws={"size": 9},  # Reduce annotation font size
        cbar_kws={'label': 'Prompt Eval Rate (tokens/s)'},
        linewidths=0.5,
        linecolor='gray'
    )

    # Add error annotations on failed cells (for highlighted version too)
    for (batch_idx, ctx_idx), error_type in errors.items():
        error_short = error_type.replace("CUDA resource allocation error", "CUDA alloc\nerror")
        error_short = error_short.replace("Parse error", "Parse\nerror")
        error_short = error_short.replace("Timeout or no output", "Timeout")

        ax.text(ctx_idx + 0.5, batch_idx + 0.5, f'✗\n{error_short}',
                ha='center', va='center', color='red', fontsize=8, fontweight='bold')

        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((ctx_idx, batch_idx), 1, 1,
                               fill=False, edgecolor='red', lw=2, linestyle='--'))

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

    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {OUTPUT_FILE}")

    return max_rate, best_ctx if len(max_pos[0]) > 0 else None, best_batch if len(max_pos[0]) > 0 else None

def print_statistics(data, num_ctx_values, num_batch_values):
    """Print summary statistics"""
    # Count successful tests (not NaN and not -1)
    successful_data = data[(~np.isnan(data)) & (data != -1)]
    # Count failed tests (marked as -1)
    failed_count = np.sum(data == -1)
    # Count untested
    total_possible = len(num_ctx_values) * len(num_batch_values)
    tested_count = np.sum(~np.isnan(data))
    untested_count = total_possible - tested_count

    print("\n" + "=" * 80)
    print("BENCHMARK STATISTICS")
    print("=" * 80)
    print(f"Total possible combinations: {total_possible}")
    print(f"Successful tests: {len(successful_data)}")
    print(f"Failed tests: {failed_count}")
    print(f"Not yet tested: {untested_count}")
    print(f"Coverage: {len(num_ctx_values)} num_ctx values × {len(num_batch_values)} num_batch values")
    print()

    if len(successful_data) > 0:
        print(f"Max prompt eval rate: {np.max(successful_data):.2f} tokens/s")
        print(f"Min prompt eval rate: {np.min(successful_data):.2f} tokens/s")
        print(f"Mean prompt eval rate: {np.mean(successful_data):.2f} tokens/s")
        print(f"Median prompt eval rate: {np.median(successful_data):.2f} tokens/s")
    else:
        print("No successful tests to compute statistics")

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
    data, num_ctx_values, num_batch_values, errors = create_heatmap_data(results)

    # Save CSV
    print("Saving CSV table...")
    save_csv(data, num_ctx_values, num_batch_values)
    print()

    # Plot heatmap
    print("Generating heatmap...")
    max_rate, best_ctx, best_batch = plot_heatmap(data, num_ctx_values, num_batch_values, results_data.get("metadata", {}), errors)

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
