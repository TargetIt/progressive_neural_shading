# Phase 8.0: ReLU Activation — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_03 (前半部分)
> **前置 Phase**: Phase 7.1

## 1. Introduction

Phase 8.0 深入研究 ReLU (Rectified Linear Unit) 激活函数的特性:
稀疏性、dead neurons、梯度特性。

```
ReLU(x) = max(0, x)
ReLU'(x) = 1 if x > 0 else 0
```

## 2. Key Properties

- **稀疏性**: 负值输入 → 输出 0 → 稀疏激活
- **Dead neurons**: 如果所有输入 < 0，梯度 = 0 → 神经元永不更新
- **非饱和**: 正值区域梯度恒为 1 → 缓解 vanishing gradient

## 3. Comparison

| Aspect | Phase 7.1 | Phase 8.0 |
|--------|-----------|-----------|
| Activation | ReLU (default) | ReLU analysis |
| Focus | Training pipeline | Activation behavior |
| Visualization | 3-panel | Activation heatmap |

## 4. Known Issues

- ReLU 的 dead neuron 问题 → Phase 8.1 探索 LeakyReLU, GELU 等变体
