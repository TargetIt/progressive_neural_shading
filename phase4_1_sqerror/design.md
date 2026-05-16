# Phase 4.1: Squared Error (MSE) — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_04_loss
> **前置 Phase**: Phase 4.0

## 1. Introduction

Phase 4.1 从 Phase 4.0 的绝对误差升级为平方误差 (squared error)。
平方误差是 MSE (Mean Squared Error) 的基础构建块，也是后续训练的目标函数。

```
sq_error(pixel) = (render(pixel) - reference)^2
MSE = mean(sq_error over all pixels)
```

## 2. Core Algorithm

```slang
float3 sq_error(int2 pixel, float3 reference, MaterialParameters mat, float3 L, float3 V) {
    float3 color = render(pixel, mat, L, V);
    float3 error = color - reference;
    return error * error;  // squared error per channel
}
```

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_4_1_sqerror.slang` | MaterialParameters + render() + sq_error() |
| `step_4_1_sqerror.py` | MSE 计算 + 3-panel display |
| `trace.py` | Tensor stats + compute_mse() |

## 4. Comparison

| Aspect | Phase 4.0 | Phase 4.1 |
|--------|-----------|-----------|
| Error type | |pred - ref| | (pred - ref)^2 |
| Console output | None | MSE printed |
| 可微分性 | 不可微 (abs) | 可微 (sq) → 训练关键 |
| Error 可视化 | 灰度 abs | per-channel squared |

## 5. Known Issues

- sq_error 返回 per-channel RGB，非标量 → Phase 4.2 可视化 loss landscape
- 对比的是不同光源角度，非纹理参数 → Phase 5 将对纹理参数优化
