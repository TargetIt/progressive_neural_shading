# Phase 5.2: Adam Optimizer

## Quick Start

```bash
python src/step_5_2_adam.py
```

观察 loss 随优化步数下降。3-panel: Ref / Pred / Loss。按 ESC 退出。

## What This Phase Teaches

- Adam optimizer: momentum + adaptive learning rates
- Training loop: gradient → Adam step → parameter update
- Loss monitoring: 观察 loss 如何随时间收敛

## New in Phase 5.2

- **Adam update**: m/v moments, bias correction
- **Iterative optimization**: 10 steps per frame
- **optimizer_step**: GPU 实现的参数更新

## Diff from Phase 5.1

| Phase 5.1 | Phase 5.2 |
|-----------|-----------|
| Static params | Optimized params |
| Gradient only | Gradient → Update |
| Loss constant | Loss decreasing |


## Using trace.py

```python
from trace import tensor_stats, print_stats, adam_stats

# 报告 Adam 优化进度
stats = adam_stats(step=100, loss=loss_val, lr=0.002)
print(f"Step {stats['step']} | Loss: {stats['loss']:.6f}")
```
