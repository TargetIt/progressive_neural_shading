# Phase 8.1: Activation Gallery — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_03 (完整版)
> **前置 Phase**: Phase 8.0

## 1. Introduction

Phase 8.1 是 Step 8 (Better Activations) 最终形态，等价于 `network/step_03_better_activations`。
对比展示多种激活函数: ReLU, LeakyReLU, GELU, Sigmoid, Swish 等。

```
Gallery (side-by-side activation outputs):
┌──────┬──────────┬──────────┬──────────┐
│ ReLU │LeakyReLU │  GELU    │  Swish   │
└──────┴──────────┴──────────┴──────────┘
```

## 2. Activation Functions

| Function | Formula | Key Property |
|----------|---------|-------------|
| ReLU | max(0, x) | Sparse, dead neurons |
| LeakyReLU | max(ax, x), a=0.01 | No dead neurons |
| GELU | x·Φ(x) | Smooth, SOTA in transformers |
| Sigmoid | 1/(1+e^(-x)) | Saturating, [0,1] |
| Swish | x·σ(x) | Smooth, non-monotonic |

## 3. Comparison

| Aspect | Phase 8.0 | Phase 8.1 |
|--------|-----------|-----------|
| Activations | ReLU only | Gallery (5+) |
| Analysis | Deep on ReLU | Comparative |
| 参考对齐 | 中间 | 等价 step_03 |

## 4. Known Issues

- GELU/Swish 计算开销比 ReLU 大 → 性能 vs 精度权衡
- 下一 Step 9 将探索频率编码 → 用 sin/cos 改善位置信息
