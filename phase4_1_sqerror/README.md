# Phase 4.1: Squared Error (MSE)

## Quick Start

```bash
python src/step_4_1_sqerror.py
```

3-panel with squared error. Console prints MSE. 按 ESC 退出。

## What This Phase Teaches

- Squared error: (pred - ref)^2 vs absolute error |pred - ref|
- MSE (Mean Squared Error): 标准训练 loss
- Why squared error is differentiable (unlike abs at zero)

## New in Phase 4.1

- **sq_error()**: (color - reference)^2 per channel
- **MSE console output**: `float(np.mean(sq_tensor))`

## Diff from Phase 4.0

| Phase 4.0 | Phase 4.1 |
|-----------|-----------|
| abs_diff | sq_error |
| 灰度 error map | RGB per-channel error |
| No MSE | MSE in console |


## Using trace.py

```python
from trace import tensor_stats, print_stats, compute_mse

# 从平方误差 Tensor 计算 MSE
mse = compute_mse(sq_tensor)
print(f'MSE: {mse:.6f}')
```
