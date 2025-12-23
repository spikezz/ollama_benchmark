#!/usr/bin/env python3
"""Calculate test parameters"""

# Parameters
num_ctx_start = 8192
num_ctx_end = 102400
num_ctx_step = 2048

num_batch_start = 32
num_batch_end = 2176  # 大于2048
num_batch_step = 128

# Calculate ranges
num_ctx_values = list(range(num_ctx_start, num_ctx_end + 1, num_ctx_step))
num_batch_values = list(range(num_batch_start, num_batch_end + 1, num_batch_step))

# Calculate total tests
total_tests = len(num_ctx_values) * len(num_batch_values)

print(f"num_ctx range: {num_ctx_values[0]} to {num_ctx_values[-1]}")
print(f"num_ctx count: {len(num_ctx_values)}")
print(f"num_ctx values: {num_ctx_values[:5]}...{num_ctx_values[-3:]}")
print()
print(f"num_batch range: {num_batch_values[0]} to {num_batch_values[-1]}")
print(f"num_batch count: {len(num_batch_values)}")
print(f"num_batch values: {num_batch_values[:5]}...{num_batch_values[-3:]}")
print()
print(f"Total tests: {total_tests}")
print(f"num_predict: 2 (minimal generation - only testing prompt eval)")
print(f"Estimated time per test: ~2.1 minutes (model reload + inference)")
print(f"Estimated total time: {total_tests * 2.1:.0f} minutes = {total_tests * 2.1 / 60:.1f} hours = {total_tests * 2.1 / 1440:.1f} days")
