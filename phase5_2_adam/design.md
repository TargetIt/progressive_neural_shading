# Phase 5.2: Adam Optimizer — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_05_train
> **前置 Phase**: Phase 5.1

## 1. Introduction

Phase 5.2 在 AD 梯度之上实现 Adam 优化器，对材质参数进行迭代优化。
Loss 随时间下降：学习的材质参数逐渐逼近 reference。

```
AD gradient → Adam update (m, v moments) → parameter update
Loss: L = mean((pred - ref)^2)  decreasing over steps
```

## 2. Core Algorithm

### Adam Update Rule

```
m_t = beta1 * m_{t-1} + (1 - beta1) * grad
v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

### Per-frame: 10 optimization steps, low-res textures for speed

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_5_2_adam.slang` | MaterialParameters + render + loss + calculate_grad + optimizer_step |
| `step_5_2_adam.py` | Adam loop: 10 steps/frame, loss tracking |
| `trace.py` | adam_stats() |

## 4. Comparison

| Aspect | Phase 5.1 | Phase 5.2 |
|--------|-----------|-----------|
| 梯度 | Computed | Used for update |
| 参数 | Static | Optimized (changing) |
| Loss | Constant | Decreasing over time |
| Moments | None | m, v (Adam) |

## 5. Known Issues

- 10 steps/frame 较慢 → Phase 5.3 完整训练管线
- 从 constant initialization 开始优化 → 下一 Phase 加入训练循环
