# Phase 5.1: Automatic Differentiation

## Quick Start

```bash
python src/step_5_1_ad.py
```

自动微分计算 loss 对 albedo/normal/roughness 的梯度。按 ESC 退出。

## What This Phase Teaches

- Slang AD: `[Differentiable]`, `[BackwardDerivativeOf]`, `bwd_diff`
- RWTensor vs Tensor: 可读写 vs 只读
- AD vs 数值梯度: 精确、快速、可扩展

## New in Phase 5.1

- **AD 梯度计算**: 一次 backward pass 获得所有梯度
- **RWTensor**: GPU 可读写存储
- **`no_diff`**: 排除不需要梯度的参数

## Diff from Phase 5.0

| Phase 5.0 | Phase 5.1 |
|-----------|-----------|
| Numerical (finite diff) | Automatic (bwd_diff) |
| 3 renders per step | 1 forward + 1 backward |
| Gradient on roughness only | All 3 parameters |


## Using trace.py

```python
from trace import tensor_stats, print_stats, ad_gradient_stats

# 分析 AD 计算的梯度 (albedo/normal/roughness)
stats = ad_gradient_stats(albedo_grad, normal_grad, roughness_grad)
print(f"|grad_albedo|: {stats['|grad_albedo|']:.6f}")
```
