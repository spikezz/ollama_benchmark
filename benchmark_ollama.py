#!/usr/bin/env python3
"""
Ollama Benchmark Script
Benchmarks nemotron_f model with different num_ctx and num_batch combinations
"""

import subprocess
import re
import json
import time
import argparse
import yaml
from pathlib import Path
from datetime import datetime

# Configuration
MODEL_NAME = "nemotron_f"
MODELFILE_TEMPLATE_PATH = "/home/spikezz/Project/modelfile_nemotron_fast"
MODELFILE_TEMP_PATH = "/home/spikezz/Project/ollama_benchmark/modelfile_temp"
PROMPT_FILE_PATH = "/home/spikezz/Project/new_repo/p"
RESULTS_FILE = "/home/spikezz/Project/ollama_benchmark/benchmark_results.json"

# Default test parameters (can be overridden by command line arguments)
DEFAULT_CTX_START = 8192
DEFAULT_CTX_END = 102400
DEFAULT_CTX_STEP = 2048

DEFAULT_BATCH_START = 32
DEFAULT_BATCH_END = 2080
DEFAULT_BATCH_STEP = 128

def read_template():
    """Read the modelfile template"""
    with open(MODELFILE_TEMPLATE_PATH, 'r') as f:
        return f.read()

def create_modelfile(template, num_ctx, num_batch, num_predict=2):
    """Create a modelfile with specified parameters"""
    # Replace num_ctx and num_batch in template
    content = re.sub(
        r'PARAMETER num_ctx \d+',
        f'PARAMETER num_ctx {num_ctx}',
        template
    )
    content = re.sub(
        r'PARAMETER num_batch \d+',
        f'PARAMETER num_batch {num_batch}',
        content
    )

    # Add or replace num_predict parameter to minimize generation
    if 'PARAMETER num_predict' in content:
        content = re.sub(
            r'PARAMETER num_predict \d+',
            f'PARAMETER num_predict {num_predict}',
            content
        )
    else:
        # Add num_predict after num_batch
        content = re.sub(
            r'(PARAMETER num_batch \d+)',
            f'\\1\nPARAMETER num_predict {num_predict}',
            content
        )

    with open(MODELFILE_TEMP_PATH, 'w') as f:
        f.write(content)

    return MODELFILE_TEMP_PATH

def create_ollama_model(modelfile_path):
    """Create ollama model from modelfile"""
    cmd = f"ollama create {MODEL_NAME} -f {modelfile_path}"
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=120
    )
    return result.returncode == 0

def run_benchmark(prompt_file):
    """Run ollama model and capture verbose output"""
    cmd = f"cat {prompt_file} | ollama run {MODEL_NAME} --verbose 2>&1"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=360  # Allow up to 6 minutes per test (includes model reload time)
        )
        output = result.stdout + result.stderr
        return output
    except subprocess.TimeoutExpired:
        return None

def parse_prompt_eval_rate(output):
    """Parse prompt eval rate from ollama verbose output"""
    if not output:
        return None

    # Look for "prompt eval rate: X tokens/s"
    match = re.search(r'prompt eval rate:\s+([\d.]+)\s+tokens/s', output)
    if match:
        return float(match.group(1))
    return None

def load_existing_results():
    """Load existing results if they exist"""
    if Path(RESULTS_FILE).exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return {"metadata": {}, "results": []}

def save_results(results):
    """Save results to JSON file"""
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)

def load_config_from_yaml(yaml_file):
    """Load configuration from YAML file"""
    try:
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Warning: Config file '{yaml_file}' not found, using defaults")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return {}

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Benchmark ollama nemotron_f model with different num_ctx and num_batch combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default parameters
  python3 benchmark_ollama.py

  # Run with config file
  python3 benchmark_ollama.py --config config.yaml

  # Run with specific config file
  python3 benchmark_ollama.py --config configs/middle_range.yaml
        """)

    parser.add_argument('--config', type=str, default='benchmark_config.yaml',
                        help='Path to YAML config file (default: benchmark_config.yaml)')

    # These can override YAML config
    parser.add_argument('--ctx-start', type=int, default=None,
                        help='num_ctx start value (overrides config file)')
    parser.add_argument('--ctx-end', type=int, default=None,
                        help='num_ctx end value (overrides config file)')
    parser.add_argument('--ctx-step', type=int, default=None,
                        help='num_ctx step size (overrides config file)')

    parser.add_argument('--batch-start', type=int, default=None,
                        help='num_batch start value (overrides config file)')
    parser.add_argument('--batch-end', type=int, default=None,
                        help='num_batch end value (overrides config file)')
    parser.add_argument('--batch-step', type=int, default=None,
                        help='num_batch step size (overrides config file)')

    return parser.parse_args()

def main():
    """Main benchmark function"""
    # Parse command line arguments
    args = parse_arguments()

    # Load config from YAML file
    config = load_config_from_yaml(args.config)

    # Get values from config file, with defaults
    ctx_start = config.get('num_ctx', {}).get('start', DEFAULT_CTX_START)
    ctx_end = config.get('num_ctx', {}).get('end', DEFAULT_CTX_END)
    ctx_step = config.get('num_ctx', {}).get('step', DEFAULT_CTX_STEP)

    batch_start = config.get('num_batch', {}).get('start', DEFAULT_BATCH_START)
    batch_end = config.get('num_batch', {}).get('end', DEFAULT_BATCH_END)
    batch_step = config.get('num_batch', {}).get('step', DEFAULT_BATCH_STEP)

    # Get test order preference (default: column-first)
    test_row_first = config.get('test_row_first', False)

    # Command line arguments override config file
    if args.ctx_start is not None:
        ctx_start = args.ctx_start
    if args.ctx_end is not None:
        ctx_end = args.ctx_end
    if args.ctx_step is not None:
        ctx_step = args.ctx_step
    if args.batch_start is not None:
        batch_start = args.batch_start
    if args.batch_end is not None:
        batch_end = args.batch_end
    if args.batch_step is not None:
        batch_step = args.batch_step

    # Create ranges based on final values
    NUM_CTX_RANGE = range(ctx_start, ctx_end + 1, ctx_step)
    NUM_BATCH_RANGE = range(batch_start, batch_end + 1, batch_step)

    print("=" * 80)
    print("Ollama Benchmark - nemotron_f Model")
    print("=" * 80)
    print(f"num_ctx range: {NUM_CTX_RANGE.start} to {NUM_CTX_RANGE.stop-1} (step {NUM_CTX_RANGE.step})")
    print(f"num_batch range: {NUM_BATCH_RANGE.start} to {NUM_BATCH_RANGE.stop-1} (step {NUM_BATCH_RANGE.step})")

    num_ctx_count = len(list(NUM_CTX_RANGE))
    num_batch_count = len(list(NUM_BATCH_RANGE))
    total_tests = num_ctx_count * num_batch_count

    print(f"Total tests: {total_tests}")
    print(f"num_predict: 2 (minimal generation to test prompt eval only)")
    if test_row_first:
        print(f"Test order: Row-first (complete horizontal rows - fixed batch, varying ctx)")
    else:
        print(f"Test order: Column-first (complete vertical columns - fixed ctx, varying batch)")
    print(f"Estimated time per test: ~2.1 minutes (model reload + inference)")
    print(f"Estimated total time: ~{total_tests * 2.1 / 60:.1f} hours (~{total_tests * 2.1 / 1440:.1f} days)")
    print("=" * 80)
    print()

    # Load existing results
    results_data = load_existing_results()
    results_data["metadata"] = {
        "start_time": datetime.now().isoformat(),
        "num_ctx_range": f"{NUM_CTX_RANGE.start}-{NUM_CTX_RANGE.stop-1}:{NUM_CTX_RANGE.step}",
        "num_batch_range": f"{NUM_BATCH_RANGE.start}-{NUM_BATCH_RANGE.stop-1}:{NUM_BATCH_RANGE.step}",
        "total_tests": total_tests
    }

    # Read template
    template = read_template()

    # Track completed tests
    completed = set()
    for r in results_data.get("results", []):
        completed.add((r["num_ctx"], r["num_batch"]))

    # Run benchmark for each combination
    test_num = 0
    start_time = time.time()

    # Choose loop order based on test_row_first flag
    if test_row_first:
        # Row-first: iterate batch (rows) in outer loop, ctx (columns) in inner loop
        outer_range = NUM_BATCH_RANGE
        inner_range = NUM_CTX_RANGE
        outer_is_batch = True
    else:
        # Column-first: iterate ctx (columns) in outer loop, batch (rows) in inner loop
        outer_range = NUM_CTX_RANGE
        inner_range = NUM_BATCH_RANGE
        outer_is_batch = False

    for outer_val in outer_range:
        for inner_val in inner_range:
            # Assign to appropriate variables based on loop order
            if outer_is_batch:
                num_batch = outer_val
                num_ctx = inner_val
            else:
                num_ctx = outer_val
                num_batch = inner_val
            test_num += 1

            # Skip if already completed
            if (num_ctx, num_batch) in completed:
                print(f"[{test_num}/{total_tests}] Skipping num_ctx={num_ctx}, num_batch={num_batch} (already completed)")
                continue

            elapsed = time.time() - start_time
            avg_time_per_test = elapsed / max(1, test_num - len(completed) - 1)
            remaining_tests = total_tests - test_num
            eta_seconds = avg_time_per_test * remaining_tests
            eta_hours = eta_seconds / 3600

            print(f"\n[{test_num}/{total_tests}] Testing num_ctx={num_ctx}, num_batch={num_batch}")
            print(f"  ETA: {eta_hours:.2f} hours ({eta_seconds/60:.1f} minutes)")

            # Create modelfile
            print("  Creating modelfile...")
            modelfile_path = create_modelfile(template, num_ctx, num_batch)

            # Create model
            print("  Creating ollama model...")
            if not create_ollama_model(modelfile_path):
                print("  ERROR: Failed to create model")
                results_data["results"].append({
                    "num_ctx": num_ctx,
                    "num_batch": num_batch,
                    "prompt_eval_rate": None,
                    "error": "Failed to create model"
                })
                save_results(results_data)
                continue

            # Run benchmark
            print("  Running benchmark...")
            output = run_benchmark(PROMPT_FILE_PATH)

            if output is None:
                print("  ERROR: Benchmark timed out")
                results_data["results"].append({
                    "num_ctx": num_ctx,
                    "num_batch": num_batch,
                    "prompt_eval_rate": None,
                    "error": "Timeout"
                })
                save_results(results_data)
                continue

            # Parse result
            rate = parse_prompt_eval_rate(output)

            if rate is not None:
                print(f"  ✓ prompt eval rate: {rate:.2f} tokens/s")
                results_data["results"].append({
                    "num_ctx": num_ctx,
                    "num_batch": num_batch,
                    "prompt_eval_rate": rate,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                print("  ERROR: Could not parse prompt eval rate")
                if output:
                    # Print last 500 chars of output to diagnose the issue
                    print(f"  Output (last 500 chars): {output[-500:]}")
                    # Check for common error patterns
                    if "out of memory" in output.lower() or "oom" in output.lower():
                        error_type = "CUDA OOM"
                    elif "cuda error" in output.lower() or "resource allocation failed" in output.lower():
                        error_type = "CUDA resource allocation error"
                    elif "error" in output.lower():
                        error_type = "Ollama error"
                    else:
                        error_type = "Parse error"
                else:
                    error_type = "Timeout or no output"
                    print(f"  No output received (likely timeout)")

                results_data["results"].append({
                    "num_ctx": num_ctx,
                    "num_batch": num_batch,
                    "prompt_eval_rate": None,
                    "error": error_type
                })

            # Save results after each test
            save_results(results_data)

    # Final summary
    results_data["metadata"]["end_time"] = datetime.now().isoformat()
    results_data["metadata"]["total_duration_seconds"] = time.time() - start_time
    save_results(results_data)

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print(f"Results saved to: {RESULTS_FILE}")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print("=" * 80)

if __name__ == "__main__":
    main()
