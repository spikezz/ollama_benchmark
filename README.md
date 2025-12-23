# Ollama nemotron_f Benchmark

自动化测试脚本，用于找到ollama nemotron_f模型的最优num_ctx和num_batch组合。

## 测试参数

- **num_ctx**: 8192 到 102400，步长 2048（共47个值）
- **num_batch**: 32 到 2080，步长 128（共17个值）
- **num_predict**: 2（最小生成量，只测试prompt eval速度）
- **总测试次数**: 799次（47 × 17）
- **预计时间**: 约28小时（1.2天）- 每次测试约2.1分钟（实测数据）

## 目标

测试不同参数组合下的 **prompt eval rate (tokens/s)**，找到最快的配置。

## 文件说明

- `calculate_params.py` - 计算测试参数和预计时间
- `benchmark_ollama.py` - 主测试脚本
- `generate_heatmap.py` - 生成热力图和统计数据
- `requirements.txt` - Python依赖包
- `benchmark_results.json` - 测试结果（运行后生成）
- `heatmap.png` - 热力图（运行后生成）
- `heatmap_highlighted.png` - 突出显示最优值的热力图（运行后生成）
- `results_table.csv` - 结果表格（运行后生成）

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 查看测试参数

```bash
python3 calculate_params.py
```

### 3. 运行基准测试

```bash
python3 benchmark_ollama.py
```

测试过程中：
- 每次测试会自动创建新的modelfile
- 使用 `/home/spikezz/Project/new_repo/p` 文件作为提示词
- 结果会实时保存到 `benchmark_results.json`
- 如果中断，重新运行会跳过已完成的测试

### 4. 生成热力图

```bash
python3 generate_heatmap.py
```

这将生成：
- `heatmap.png` - 完整热力图
- `heatmap_highlighted.png` - 突出显示最优值的热力图
- `results_table.csv` - CSV格式的结果表格
- 控制台输出统计信息和最优配置

## 测试工作原理

1. 脚本读取modelfile模板 (`/home/spikezz/Project/modelfile_nemotron_fast`)
2. 对于每个num_ctx和num_batch组合（47×17=799次）：
   - 修改modelfile中的参数（num_ctx, num_batch, num_predict=2）
   - 运行 `ollama create nemotron_f -f modelfile_temp` 创建新模型
   - 模型重新加载到5张GPU，分配KV cache和context（约1.5-2分钟）
   - 用verbose模式运行模型: `cat p | ollama run nemotron_f --verbose`（约30-40秒）
   - 解析输出中的 "prompt eval rate" 值
   - 立即保存结果到 `benchmark_results.json`
3. 完成后运行 `python3 generate_heatmap.py` 生成热力图可视化

**实际测试时间分析（基于实测数据）：**
- 每次测试平均耗时：**128秒（2.1分钟）**
- 模型加载 + KV cache分配：约90秒
- 推理执行（num_predict=2）：约38秒
- 总计799次测试：约28小时（1.2天）

**为什么每次测试需要2分钟？**
- 改变num_ctx和num_batch需要重新配置模型的内存分配
- 必须重新加载模型权重到5张GPU
- 重新分配KV cache和context缓冲区
- 这个过程无法通过命令行参数避免，是硬件限制

## 结果输出

测试完成后，你将获得：

1. **最优配置** - num_ctx和num_batch的最佳组合及其prompt eval rate
2. **热力图** - 可视化所有47×17（799个）组合的性能
   - `heatmap.png` - 完整热力图
   - `heatmap_highlighted.png` - 标注最优值的热力图
3. **统计数据** - 最大值、最小值、平均值、中位数、标准差等
4. **CSV表格** - 可用于Excel或进一步分析（`results_table.csv`）
5. **JSON数据** - 完整的测试数据（`benchmark_results.json`）

**测试范围：**
- num_ctx: 8192, 10240, 12288, ..., 100352, 102400
- num_batch: 32, 160, 288, 416, 544, 672, 800, 928, 1056, 1184, 1312, 1440, 1568, 1696, 1824, 1952, 2080

## 查看中间结果

测试进行中也可以随时查看当前结果：

```bash
# 查看已完成的测试数量和进度
python3 -c "import json; data=json.load(open('benchmark_results.json')); print(f'已完成: {len(data[\"results\"])}/799 ({len(data[\"results\"])/799*100:.1f}%)')"

# 查看最新几个测试结果
python3 << 'EOF'
import json
data = json.load(open('benchmark_results.json'))
print("最新5个测试:")
for r in data['results'][-5:]:
    rate = r.get('prompt_eval_rate', 'N/A')
    print(f"  ctx={r['num_ctx']}, batch={r['num_batch']}, rate={rate}")
EOF

# 生成当前的热力图（基于已完成的测试）
python3 generate_heatmap.py

# 查看实时测试输出
tail -f /tmp/claude/-home-spikezz/tasks/[task_id].output

# 分析测试速度
python3 << 'EOF'
import json
from datetime import datetime
data = json.load(open('benchmark_results.json'))
results = [r for r in data['results'] if 'timestamp' in r]
if len(results) >= 2:
    times = [datetime.fromisoformat(r['timestamp']) for r in results]
    avg = sum((times[i]-times[i-1]).total_seconds() for i in range(1,len(times)))/(len(times)-1)
    print(f"平均测试时间: {avg:.0f}秒 ({avg/60:.1f}分钟)")
    remaining = 799 - len(data['results'])
    eta_hours = remaining * avg / 3600
    print(f"剩余{remaining}个测试，预计还需: {eta_hours:.1f}小时")
EOF
```

## 注意事项

- 测试需要约28小时完成（每次测试约2.1分钟，实测数据）
- 每次测试包含：模型重载到GPU、KV cache分配、推理等过程
- 确保ollama服务正在运行
- 测试期间请勿手动操作ollama
- 结果会自动保存到 `benchmark_results.json`，可以随时中断和恢复
- 每次测试超时设置为360秒（6分钟）
- num_predict设置为2，模型几乎不生成内容，只测试提示词处理速度
- 建议在空闲时段运行（如夜间），脚本支持断点续传
- 每次改变num_ctx/num_batch都需要重新加载模型，这是耗时的主要原因
- 实际测试显示每次约128秒（2.1分钟），比初始估计的5分钟快得多

## 示例输出

```
[1/799] Testing num_ctx=8192, num_batch=32
  ETA: 28.0 hours
  Creating modelfile...
  Creating ollama model...
  Running benchmark...
  ✓ prompt eval rate: 78.45 tokens/s

[2/799] Testing num_ctx=8192, num_batch=160
  ETA: 27.9 hours
  Creating modelfile...
  Creating ollama model...
  Running benchmark...
  ✓ prompt eval rate: 82.31 tokens/s

...

[13/799] Testing num_ctx=20480, num_batch=1088
  ETA: 27.5 hours
  Creating modelfile...
  Creating ollama model...
  Running benchmark...
  ✓ prompt eval rate: 85.47 tokens/s
```

测试完成后生成热力图：
```
OPTIMAL CONFIGURATION
num_ctx: 16384
num_batch: 832
Prompt eval rate: 85.73 tokens/s

(最终结果会基于全部799个测试确定)
```
