# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an Ollama benchmark tool designed to find optimal `num_ctx` and `num_batch` parameter combinations for the `nemotron_f` model by measuring **prompt eval rate (tokens/s)**. Tests are long-running (20+ hours) and support checkpoint/resume functionality.

## Key Commands

### Install Dependencies
```bash
pip install -r requirements.txt
```

Required packages: numpy (<2.0 recommended), matplotlib, seaborn, pandas, pyyaml

### Run Benchmark
```bash
# Full benchmark (default config)
python3 benchmark_ollama.py

# Quick test with custom config
python3 benchmark_ollama.py --config config_quick_test.yaml

# Run in background (recommended for long tests)
python3 -u benchmark_ollama.py --config config_quick_test.yaml 2>&1 &
```

### Generate Visualization
```bash
python3 generate_heatmap.py
```

Outputs: `heatmap_highlighted.png` (with best value highlighted), `results_table.csv`

### Calculate Parameters
```bash
python3 calculate_params.py
```

### Monitor Progress
```bash
# View completion rate
python3 -c "import json; data=json.load(open('benchmark_results.json')); print(f'Completed: {len(data[\"results\"])}')"

# Generate intermediate heatmap while test is running
python3 generate_heatmap.py
```

## Architecture

### Core Test Loop Structure

**CONFIGURABLE**: The benchmark supports two testing orders via `test_row_first` flag in YAML config:

**Column-first (test_row_first: false, default)**:
```python
for num_ctx in NUM_CTX_RANGE:      # Outer loop: iterate through ctx values
    for num_batch in NUM_BATCH_RANGE:  # Inner loop: iterate through batch values
```
Tests all `num_batch` values for each `num_ctx` value before moving to the next `num_ctx` (completes vertical columns in the heatmap).

**Row-first (test_row_first: true)**:
```python
for num_batch in NUM_BATCH_RANGE:   # Outer loop: iterate through batch values
    for num_ctx in NUM_CTX_RANGE:   # Inner loop: iterate through ctx values
```
Tests all `num_ctx` values for each `num_batch` value before moving to the next `num_batch` (completes horizontal rows in the heatmap).

The loop order is chosen at runtime based on the config flag.

### Test Execution Flow

Each test iteration:
1. Creates a temporary modelfile with updated `num_ctx`, `num_batch`, `num_predict=2`
2. Runs `ollama create nemotron_f -f modelfile_temp` (triggers full model reload to GPU)
3. Executes `cat p | ollama run nemotron_f --verbose`
4. Parses "prompt eval rate" from verbose output
5. Saves result to `benchmark_results.json` immediately
6. Skips already-completed tests on restart (checkpoint/resume)

**Each test takes ~2.1 minutes**: Model reload to 5 GPUs + KV cache allocation (~90s) + inference (~30s)

### YAML Configuration System

Test ranges are configured via YAML files rather than hardcoded values:

- `benchmark_config.yaml` - Default full-range configuration
- `config_quick_test.yaml` - Smaller subset for validation

Command-line arguments can override YAML config values:
```bash
python3 benchmark_ollama.py --config myconfig.yaml --ctx-start 32768 --batch-end 4096
```

Config structure:
```yaml
num_ctx:
  start: 32768
  end: 110592
  step: 2048

num_batch:
  start: 128
  end: 2048
  step: 128

# Test iteration order
test_row_first: false  # false=column-first (default), true=row-first
```

- `test_row_first: false` (default) - Column-first: completes all batch values for each ctx (vertical columns)
- `test_row_first: true` - Row-first: completes all ctx values for each batch (horizontal rows)

### Checkpoint/Resume System

`benchmark_results.json` stores all completed tests. On restart:
- Script loads existing results into a `completed` set of `(num_ctx, num_batch)` tuples
- Skips any combination already in the set
- Continues from where it left off

This allows pausing/stopping tests without losing progress.

### Error Handling in Heatmaps

Failed tests are stored with `"prompt_eval_rate": null` and `"error": "error_type"`:
- `generate_heatmap.py` marks failed cells with `-1` internally
- Displays red ✗ with error type annotation on heatmap
- Red dashed border around failed cells
- Error types detected: "CUDA OOM", "CUDA resource allocation error", "Timeout or no output", "Parse error"

### Data Flow

```
modelfile_nemotron_fast (template)
    ↓
benchmark_ollama.py (modifies params, runs tests)
    ↓
benchmark_results.json (stores: num_ctx, num_batch, prompt_eval_rate, timestamp, error)
    ↓
generate_heatmap.py (visualizes)
    ↓
heatmap_highlighted.png, results_table.csv
```

## Important Implementation Details

### Model Reload Requirement

**Cannot be optimized**: Changing `num_ctx` or `num_batch` requires full model reload to GPU, KV cache reallocation, and context buffer setup. This is a hardware limitation, not a software inefficiency.

### Heatmap Figure Sizing

Dynamically sized based on data dimensions to prevent number overlap:
```python
fig_width = max(20, len(num_ctx_values) * 0.8)   # Currently 0.8 inches per column
fig_height = max(6, len(num_batch_values) * 0.4) # Currently 0.4 inches per row
```

User has tuned these values for optimal display. Do not change without explicit request.

### num_predict=2 Setting

Tests use `num_predict=2` (minimal token generation) to isolate prompt processing performance from generation performance. This is intentional—prompt eval rate is the target metric.

## External Dependencies

- **Modelfile template**: `/home/spikezz/Project/modelfile_nemotron_fast`
- **Prompt file**: `/home/spikezz/Project/new_repo/p`
- **Ollama service**: Must be running with `nemotron_f` base model available

## Configuration vs Code Modification

**Always prefer YAML config files over modifying source code** for test range changes. User explicitly requested this approach. Create new YAML configs for different test scenarios rather than editing `benchmark_ollama.py`.

## Common Workflows

### Retest Failed Configurations
Remove failed entries from `benchmark_results.json` and rerun:
```bash
python3 << 'EOF'
import json
with open('benchmark_results.json', 'r') as f:
    data = json.load(f)
data['results'] = [r for r in data['results']
                   if not (r['num_ctx'] == 92160 and r['num_batch'] == 128)]
with open('benchmark_results.json', 'w') as f:
    json.dump(data, f, indent=2)
EOF
python3 benchmark_ollama.py --config config_quick_test.yaml
```

### Backup and Restore Results
```bash
# Backup before switching configs
cp benchmark_results.json benchmark_results_backup_$(date +%Y%m%d_%H%M%S).json

# Restore backup
cp benchmark_results_backup_20251223_202917.json benchmark_results.json
```

### Switch Test Order
Edit config file to change `test_row_first`:
- `false` - Column-first (default): complete vertical columns
- `true` - Row-first: complete horizontal rows
