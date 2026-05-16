# Phase 10.1: Trainable Latent Texture — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_05
> **前置 Phase**: Phase 10.0

## 1. Introduction

Phase 10.1 将 Phase 10.0 的固定 latent texture 改为可训练参数。
Latent texture 通过反向传播与 MLP 权重一起优化。

```
Trainable Latent Texture (H×W×C)
    ↓ bilinear sample + bwd_diff
Feature vector → MLP → RGB
    ↓
Loss → backward → gradient to latent + MLP weights
```

## 2. Key Change

- Phase 10.0: Latent is `Tensor` (read-only)
- Phase 10.1: Latent is `RWTensor` with gradient accumulation

## 3. Comparison

| Aspect | Phase 10.0 | Phase 10.1 |
|--------|-----------|-----------|
| Latent | Fixed random | Trainable |
| Gradient | Only MLP | MLP + latent |
| Optimization | Partial | End-to-end |

## 4. Known Issues

- 需要完整的 latent backward pass → Phase 10.2 完整管线 (latent + MLP + Adam)
