# Phase 10.2: Full Pipeline — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_05 (完整版)
> **前置 Phase**: Phase 10.1

## 1. Introduction

Phase 10.2 是 Step 10 (Latent Texture) 最终形态，等价于 `network/step_05_latent_texture`。
完整的神经网络纹理管线: Latent Texture → MLP → Material Params → BRDF → Loss。

```
Latent Texture (H×W×C, trainable)
    ↓ bilinear sample
Feature → MLP → Albedo + Normal + Roughness
    ↓
Disney BRDF → Render → SSAA reference → Loss
    ↓
bwd_diff → Adam(MLP + Latent)
```

## 2. Full Pipeline Components

- **Latent Texture**: 可训练的空间特征图
- **MLP**: 多层网络 (latent features → material params)
- **BRDF**: Disney BRDF 可微渲染
- **Loss**: SSAA reference vs prediction squared error
- **Optimizer**: Adam with LR decay

## 3. Comparison

| Aspect | Phase 10.1 | Phase 10.2 |
|--------|-----------|-----------|
| Pipeline | Latent + MLP only | Latent + MLP + BRDF |
| Output | RGB directly | Material → BRDF → RGB |
| 参考对齐 | 中间 | 等价 step_05 |

## 4. Known Issues

- 计算量大 (BRDF per pixel + backward) → Step 11 硬件加速
