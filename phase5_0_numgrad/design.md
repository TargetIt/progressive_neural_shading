# Phase 5.0: Numerical Gradient — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_05_train (前半部分)
> **前置 Phase**: Phase 4.2

## 1. Introduction

Phase 5.0 用中央有限差分 (central finite differences) 计算 loss 对 roughness 的数值梯度。
这是理解梯度下降训练的第一步: 了解"如何计算梯度"而不涉及自动微分。

```
dL/d(roughness) ≈ (L(roughness + h) - L(roughness - h)) / (2h)
```

## 2. Core Algorithm

### Central finite differences

```python
h = 0.01
roughness_plus = roughness + h
loss_plus = render_and_loss(roughness_plus)
roughness_minus = roughness - h
loss_minus = render_and_loss(roughness_minus)
gradient = (loss_plus - loss_minus) / (2 * h)
```

### Why numerical gradient?

- 概念简单: 只需要前向 pass (render + loss)
- 缺点: 计算慢 (2 次前向 pass 每个参数)，精度受 h 选择影响
- 为 Phase 5.1 的自动微分提供对比 reference

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_5_0_numgrad.slang` | MaterialParameters + render() + loss() |
| `step_5_0_numgrad.py` | Central finite diff on roughness, 梯度可视化 |
| `trace.py` | gradient_stats() |

## 4. Display

- Left: reference render (full resolution)
- Right: gradient magnitude heatmap (grayscale)

## 5. Comparison

| Aspect | Phase 4.2 | Phase 5.0 |
|--------|-----------|-----------|
| Loss | Visualized | Used for gradient |
| 梯度 | None | Numerical (finite diff) |
| 计算次数 | 2 renders | 3 renders (ref + plus + minus) |

## 6. Known Issues

- 性能: 每次梯度计算需 2 次额外渲染
- 精度: h 选 0.01 是折中 (太小→浮点误差, 太大→近似误差)
- 下一 Phase 5.1 将引入自动微分 (AD)
