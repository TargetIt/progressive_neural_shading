# Phase 7.1: Deep Network Training — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_02 (完整版)
> **前置 Phase**: Phase 7.0

## 1. Introduction

Phase 7.1 是 Step 7 (Multiple Layers) 最终形态，等价于 `network/step_02_multiple_layers`。
3 层 MLP: 2→32→32→3，完整训练管线。

```
Input (2D uv)
    ↓
layer0: W₀(32×2) + b₀(32) → ReLU
    ↓
layer1: W₁(32×32) + b₁(32) → ReLU
    ↓
layer2: W₂(3×32) + b₂(3)
    ↓
Output (RGB)
```

## 2. Key Features

- **3 层深度**: 比 Phase 7.0 多一个隐藏层
- **32 隐藏神经元**: 比 Phase 7.0 的 16 更大容量
- **完整训练**: Adam + atomic gradient + LCG sampling

## 3. Comparison

| Aspect | Phase 7.0 | Phase 7.1 |
|--------|-----------|-----------|
| Layers | 2 | 3 |
| Hidden | 16 | 32×2 |
| Training | Forward only | Full training |
| 参考对齐 | 中间 | 等价 step_02 |

## 4. Known Issues

- ReLU only → Step 8 探索更好的激活函数
- 固定结构 (32,32) → Step 9 学习配置技术
