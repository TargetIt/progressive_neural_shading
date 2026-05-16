# Phase 10.0: Fixed Latent Texture — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_05 (前半部分)
> **前置 Phase**: Phase 9.1

## 1. Introduction

Phase 10.0 用固定 latent texture 替代频率编码作为网络输入。
Latent texture 是一个低分辨率的特征图，通过双线性采样获取每像素的特征向量。

```
Latent Texture (H×W×C)
    ↓ bilinear sample at UV
Feature vector (C dim)
    ↓ MLP
Output (RGB)
```

## 2. Key Concept

- **Latent texture**: 替代 sin/cos encoding，存储空间变化的特征
- **Fixed (不可训练)**: 使用固定随机值，验证 latent texture 概念
- **Bilinear sampling**: GPU 硬件加速的纹理采样

## 3. Comparison

| Aspect | Phase 9.1 | Phase 10.0 |
|--------|-----------|-----------|
| Input | sin/cos encoded UV | Sampled latent texture |
| Dimensionality | 4N (exploding) | C (compact) |
| Trainable | No params | Latent is fixed |

## 4. Known Issues

- 固定 latent → 表达能力受限 → Phase 10.1 使 latent 可训练
