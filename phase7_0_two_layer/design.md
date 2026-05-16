# Phase 7.0: Two-Layer Forward — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_02 (前半部分)
> **前置 Phase**: Phase 6.2

## 1. Introduction

Phase 7.0 将 Phase 6.2 的单层 MLP 扩展为双层网络:
`uv(2) → layer0(16) → ReLU → layer1(3) → output`

```
Input (2D uv)
    ↓
layer0: W₀(16×2) + b₀(16)
    ↓
ReLU
    ↓
layer1: W₁(3×16) + b₁(3)
    ↓
Output (RGB)
```

## 2. Key Architecture

- **layer0**: 2→16 (hidden layer, with ReLU)
- **layer1**: 16→3 (output layer, no activation = linear)
- **NetworkParameters**: 复用 Phase 6.2 的参数化层

## 3. Comparison

| Aspect | Phase 6.2 | Phase 7.0 |
|--------|-----------|-----------|
| Layers | 1 (3×2) | 2 (16×2 + 3×16) |
| Hidden dim | None | 16 |
| Forward only | Training | Forward pass only (no training yet) |

## 4. Known Issues

- 只做前向，没有训练 → Phase 7.1 添加多层训练
- 硬编码 16 神经元 → 后续 Phase 学习配置隐藏层维度
