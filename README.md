# Ollama nemotron_f Benchmark

自动化测试脚本，用于找到Ollama nemotron_f模型的最优num_ctx和num_batch组合。

## 项目概述

通过系统化测试不同的 `num_ctx` 和 `num_batch` 参数组合，测量 **prompt eval rate (tokens/s)**，找到最快的配置。支持长时间测试的断点续传和实时进度监控。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

需要的包：
- numpy (建议 <2.0，避免与matplotlib兼容性问题)
- matplotlib
- seaborn
- pandas
- pyyaml

### 2. 配置测试范围

编辑YAML配置文件来设置测试参数：

**默认配置** (`benchmark_config.yaml`):
```yaml
num_ctx:
  start: 8192
  end: 102400
  step: 2048

num_batch:
  start: 32
  end: 2080
  step: 128

test_row_first: false  # Column-first (完成垂直列)
```
- 总测试数: 47 × 17 = 799次
- 预计时间: ~28小时

**快速测试** (`config_quick_test.yaml`):
```yaml
num_ctx:
  start: 32768
  end: 110592
  step: 2048

num_batch:
  start: 128
  end: 2048
  step: 128

test_row_first: false  # Column-first (完成垂直列)
```
- 总测试数: 39 × 16 = 624次
- 预计时间: ~21.8小时

或创建自己的配置文件用于不同测试场景。

### 3. 运行基准测试

```bash
# 使用默认配置
python3 benchmark_ollama.py

# 使用指定配置
python3 benchmark_ollama.py --config config_quick_test.yaml

# 后台运行（推荐用于长时间测试）
python3 -u benchmark_ollama.py --config config_quick_test.yaml 2>&1 &
```

**命令行参数覆盖**:
```bash
python3 benchmark_ollama.py --config myconfig.yaml \
    --ctx-start 32768 --ctx-end 81920 --ctx-step 2048 \
    --batch-start 128 --batch-end 2048 --batch-step 128
```

### 4. 生成热力图

```bash
python3 generate_heatmap.py
```

这将生成：
- `heatmap_highlighted.png` - 突出显示最优值的热力图
- `results_table.csv` - CSV格式的结果表格
- 控制台输出统计信息和最优配置

## 文件说明

### 核心脚本
- `benchmark_ollama.py` - 主测试脚本，支持YAML配置和断点续传
- `generate_heatmap.py` - 生成热力图可视化，显示失败测试并标注错误原因
- `calculate_params.py` - 计算测试参数和预计时间

### 配置文件
- `benchmark_config.yaml` - 默认全范围配置
- `config_quick_test.yaml` - 快速测试配置

### 生成文件
- `benchmark_results.json` - 测试结果（自动保存，用于断点续传）
- `benchmark_results_backup_*.json` - 结果备份（手动创建）
- `heatmap_highlighted.png` - 热力图可视化（带最优值标注）
- `results_table.csv` - 结果表格
- `modelfile_temp` - 临时modelfile（测试期间生成）

### 其他
- `requirements.txt` - Python依赖包
- `README.md` - 本文件
- `CLAUDE.md` - 代码库架构说明（供Claude Code使用）

## 测试工作原理

### 执行流程

每个测试组合都会：

1. **创建modelfile** - 从模板 `/home/spikezz/Project/modelfile_nemotron_fast` 读取，修改以下参数：
   - `PARAMETER num_ctx <value>`
   - `PARAMETER num_batch <value>`
   - `PARAMETER num_predict 2` (最小生成量)

2. **重新创建模型** - `ollama create nemotron_f -f modelfile_temp`
   - 触发模型重新加载到5张GPU
   - 重新分配KV cache和context缓冲区
   - 耗时约90秒

3. **运行推理** - `cat /home/spikezz/Project/new_repo/p | ollama run nemotron_f --verbose`
   - 使用verbose模式捕获性能指标
   - 耗时约30-40秒

4. **解析结果** - 从输出中提取 "prompt eval rate: X tokens/s"

5. **保存结果** - 立即追加到 `benchmark_results.json`

6. **断点续传** - 重启时自动跳过已完成的测试

### 测试顺序

可以通过配置文件中的 `test_row_first` 标志控制测试顺序：

**列优先（test_row_first: false，默认）**：
- 先测试 `num_ctx=<start>` 的所有 `num_batch` 值（完成一列）
- 再测试 `num_ctx=<start+step>` 的所有 `num_batch` 值
- 以此类推

例如对于配置 `ctx: 32768-36864:2048, batch: 128-384:128, test_row_first: false`:
```
[1/9] ctx=32768, batch=128
[2/9] ctx=32768, batch=256
[3/9] ctx=32768, batch=384  ← 完成第一列（ctx=32768）
[4/9] ctx=34816, batch=128
[5/9] ctx=34816, batch=256
[6/9] ctx=34816, batch=384  ← 完成第二列（ctx=34816）
...
```

**行优先（test_row_first: true）**：
- 先测试 `num_batch=<start>` 的所有 `num_ctx` 值（完成一行）
- 再测试 `num_batch=<start+step>` 的所有 `num_ctx` 值
- 以此类推

例如对于配置 `ctx: 32768-36864:2048, batch: 128-384:128, test_row_first: true`:
```
[1/9] ctx=32768, batch=128
[2/9] ctx=34816, batch=128
[3/9] ctx=36864, batch=128  ← 完成第一行（batch=128）
[4/9] ctx=32768, batch=256
[5/9] ctx=34816, batch=256
[6/9] ctx=36864, batch=256  ← 完成第二行（batch=256）
...
```

### 时间估算

**每次测试约2.1分钟** (实测数据):
- 模型加载 + KV cache分配: ~90秒
- 推理执行 (num_predict=2): ~30秒

**为什么这么慢？**
改变 `num_ctx` 或 `num_batch` 需要重新配置模型的内存分配，必须：
- 重新加载模型权重到5张GPU
- 重新分配KV cache
- 重新分配context缓冲区

这是硬件限制，无法通过命令行参数或其他方式优化。

### 错误处理

测试可能因以下原因失败：
- **CUDA OOM** - 显存不足
- **CUDA resource allocation error** - GPU资源分配失败（通常是临时性问题）
- **Timeout** - 测试超时（>6分钟）
- **Parse error** - 无法解析输出

失败的测试会：
- 保存为 `"prompt_eval_rate": null, "error": "错误类型"`
- 在热力图中用红色✗标记并显示错误原因
- 可以通过删除JSON中的条目重新测试

## 监控进度

### 查看完成情况

```bash
# 查看已完成的测试数量
python3 -c "import json; data=json.load(open('benchmark_results.json')); print(f'已完成: {len(data[\"results\"])}')"

# 查看成功/失败统计
python3 << 'EOF'
import json
data = json.load(open('benchmark_results.json'))
success = sum(1 for r in data['results'] if r.get('prompt_eval_rate') is not None)
failed = sum(1 for r in data['results'] if r.get('error'))
print(f"成功: {success}, 失败: {failed}, 总计: {len(data['results'])}")
EOF
```

### 查看最新结果

```bash
python3 << 'EOF'
import json
data = json.load(open('benchmark_results.json'))
print("最新5个测试:")
for r in data['results'][-5:]:
    rate = r.get('prompt_eval_rate', 'FAILED')
    error = r.get('error', '')
    print(f"  ctx={r['num_ctx']}, batch={r['num_batch']}, rate={rate} {error}")
EOF
```

### 生成中间热力图

测试期间可以随时生成热力图查看当前结果：

```bash
python3 generate_heatmap.py
```

### 分析测试速度

```bash
python3 << 'EOF'
import json
from datetime import datetime
data = json.load(open('benchmark_results.json'))
results = [r for r in data['results'] if 'timestamp' in r]
if len(results) >= 2:
    times = [datetime.fromisoformat(r['timestamp']) for r in results]
    avg = sum((times[i]-times[i-1]).total_seconds() for i in range(1,len(times)))/(len(times)-1)
    print(f"平均测试时间: {avg:.0f}秒 ({avg/60:.1f}分钟)")

    total_tests = data['metadata'].get('total_tests', 0)
    remaining = total_tests - len(data['results'])
    if remaining > 0:
        eta_hours = remaining * avg / 3600
        print(f"剩余{remaining}个测试，预计还需: {eta_hours:.1f}小时")
EOF
```

## 断点续传和备份

### 暂停和恢复

测试可以随时中断（Ctrl+C 或关机）：

```bash
# 停止测试
pkill -f "benchmark_ollama.py"

# 重新启动（会自动跳过已完成的测试）
python3 benchmark_ollama.py --config config_quick_test.yaml
```

### 备份结果

在切换测试配置前备份当前结果：

```bash
cp benchmark_results.json benchmark_results_backup_$(date +%Y%m%d_%H%M%S).json
```

### 恢复备份

```bash
cp benchmark_results_backup_20251223_202917.json benchmark_results.json
```

### 清空结果重新开始

```bash
# 备份旧结果
mv benchmark_results.json benchmark_results_old.json

# 或直接删除
rm benchmark_results.json
```

## 结果解读

### 热力图说明

生成的热力图显示：
- **X轴**: num_ctx 值
- **Y轴**: num_batch 值
- **颜色**: prompt eval rate (tokens/s)，越红越快
- **蓝色边框**: 最优配置
- **红色✗**: 失败的测试，标注错误类型

### 统计信息

```
================================================================================
BENCHMARK STATISTICS
================================================================================
Total possible combinations: 624
Successful tests: 56
Failed tests: 1
Not yet tested: 567
Coverage: 39 num_ctx values × 16 num_batch values

Max prompt eval rate: 121.33 tokens/s
Min prompt eval rate: 79.65 tokens/s
Mean prompt eval rate: 99.40 tokens/s
Median prompt eval rate: 84.63 tokens/s
================================================================================

================================================================================
OPTIMAL CONFIGURATION
================================================================================
num_ctx: 59392
num_batch: 128
Prompt eval rate: 121.33 tokens/s
================================================================================
```

### CSV导出

`results_table.csv` 包含完整的数据矩阵，可以：
- 在Excel中打开分析
- 导入其他可视化工具
- 进行自定义数据分析

## 注意事项

1. **确保Ollama服务运行** - 测试依赖Ollama服务
2. **不要手动操作Ollama** - 测试期间避免使用 `ollama run` 等命令
3. **测试时间很长** - 完整测试需要20-30小时，建议夜间运行
4. **自动保存** - 每次测试后立即保存结果，可以安全中断
5. **配置冲突** - 切换不同范围的配置前先备份 `benchmark_results.json`
6. **依赖文件** - 确保以下文件存在：
   - `/home/spikezz/Project/modelfile_nemotron_fast` (modelfile模板)
   - `/home/spikezz/Project/new_repo/p` (提示词文件)

## 高级用法

### 创建自定义配置

创建新的YAML配置文件测试特定范围：

```yaml
# config_high_ctx.yaml - 测试高ctx值
num_ctx:
  start: 65536
  end: 110592
  step: 4096

num_batch:
  start: 128
  end: 1024
  step: 128
```

```bash
python3 benchmark_ollama.py --config config_high_ctx.yaml
```

### 重测失败的配置

从 `benchmark_results.json` 中删除失败的条目，重新运行测试即可：

```bash
python3 << 'EOF'
import json

with open('benchmark_results.json', 'r') as f:
    data = json.load(f)

# 删除 ctx=92160, batch=128 的失败记录
data['results'] = [r for r in data['results']
                   if not (r['num_ctx'] == 92160 and r['num_batch'] == 128)]

with open('benchmark_results.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Remaining tests: {len(data['results'])}")
EOF

# 重新运行测试
python3 benchmark_ollama.py --config config_quick_test.yaml
```

### 合并多个测试结果

如果分别运行了不同范围的测试，可以合并结果：

```bash
python3 << 'EOF'
import json

# 加载多个结果文件
with open('benchmark_results_1.json', 'r') as f:
    data1 = json.load(f)
with open('benchmark_results_2.json', 'r') as f:
    data2 = json.load(f)

# 合并去重
combined = data1['results'] + data2['results']
unique = {(r['num_ctx'], r['num_batch']): r for r in combined}

# 保存合并结果
merged = {
    'metadata': data1['metadata'],
    'results': list(unique.values())
}

with open('benchmark_results_merged.json', 'w') as f:
    json.dump(merged, f, indent=2)

print(f"Merged {len(unique)} unique tests")
EOF
```

## 常见问题

**Q: 为什么每次测试需要2分钟？**

A: 改变 `num_ctx` 或 `num_batch` 需要完全重新加载模型到GPU并重新分配内存。这是硬件限制，无法优化。

**Q: 可以同时运行多个测试吗？**

A: 不建议。每次测试都会重新创建 `nemotron_f` 模型，并发运行会互相干扰。

**Q: 测试失败了怎么办？**

A: 大部分失败是临时性的（如GPU资源分配失败）。从JSON中删除失败记录，重新运行即可重测。

**Q: 如何只测试部分范围？**

A: 创建自定义YAML配置文件，或使用命令行参数覆盖范围。

**Q: 热力图数字重叠怎么办？**

A: `generate_heatmap.py` 会根据数据量自动调整图像大小。如需手动调整，修改 `fig_width` 和 `fig_height` 计算公式。

**Q: 如何找到特定ctx或batch的最优值？**

A: 查看 `results_table.csv`，在Excel中筛选特定行或列，找到该行/列的最大值。

## 示例输出

### 测试运行时

```
================================================================================
Ollama Benchmark - nemotron_f Model
================================================================================
num_ctx range: 32768 to 110592 (step 2048)
num_batch range: 128 to 2048 (step 128)
Total tests: 624
num_predict: 2 (minimal generation to test prompt eval only)
Estimated time per test: ~2.1 minutes (model reload + inference)
Estimated total time: ~21.8 hours (~0.9 days)
================================================================================

[1/624] Skipping num_ctx=32768, num_batch=128 (already completed)
[2/624] Skipping num_ctx=32768, num_batch=256 (already completed)

[3/624] Testing num_ctx=32768, num_batch=384
  ETA: 21.7 hours (1304.1 minutes)
  Creating modelfile...
  Creating ollama model...
  Running benchmark...
  ✓ prompt eval rate: 83.34 tokens/s

[4/624] Testing num_ctx=32768, num_batch=512
  ETA: 21.6 hours (1297.2 minutes)
  Creating modelfile...
  Creating ollama model...
  Running benchmark...
  ✓ prompt eval rate: 82.60 tokens/s
```

### 错误示例

```
[30/624] Testing num_ctx=92160, num_batch=128
  ETA: 20.8 hours (1249.5 minutes)
  Creating modelfile...
  Creating ollama model...
  Running benchmark...
  ERROR: Could not parse prompt eval rate
  Output (last 500 chars): Error: 500 Internal Server Error: llama runner process has terminated: CUDA error: the resource allocation failed
  current device: 4, in function cublas_handle at //ml/backend/ggml/ggml/src/ggml-cuda/common.cuh:1260
```

### 最终结果

```
================================================================================
OPTIMAL CONFIGURATION
================================================================================
num_ctx: 59392
num_batch: 128
Prompt eval rate: 121.33 tokens/s
================================================================================
```

## 项目结构

```
ollama_benchmark/
├── benchmark_ollama.py          # 主测试脚本
├── generate_heatmap.py          # 可视化生成器
├── calculate_params.py          # 参数计算工具
├── benchmark_config.yaml        # 默认配置
├── config_quick_test.yaml       # 快速测试配置
├── requirements.txt             # Python依赖
├── README.md                    # 本文件
├── CLAUDE.md                    # 代码库架构文档
├── .gitignore                   # Git忽略规则
├── benchmark_results.json       # 测试结果（生成）
├── heatmap_highlighted.png      # 热力图（生成）
└── results_table.csv            # 结果表格（生成）
```
