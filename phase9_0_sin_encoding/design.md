# Phase 9.0: Sin Encoding — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_04 (前半部分)
> **前置 Phase**: Phase 8.1

## 1. Introduction

Phase 9.0 引入 sin 位置编码 (frequency encoding)。将低维 UV 坐标通过 sin 函数映射到高维空间，使 MLP 更容易学习高频细节。

```
UV(2D) → sin(freq * UV) → higher-dim input → MLP
```

## 2. Core Algorithm

```slang
// Single frequency sin encoding
float2 encode(float2 uv, float freq) {
    return float2(sin(freq * uv.x), sin(freq * uv.y));
}
```

### Why sin encoding?
- MLP 对低频函数有 bias (spectral bias)
- Sin 编码将低频输入映射到高频空间
- 使网络更容易拟合纹理细节

## 3. Comparison

| Aspect | Phase 8.1 | Phase 9.0 |
|--------|-----------|-----------|
| Input | Raw UV | sin(freq * UV) |
| Focus | Activations | Positional encoding |
| Detail | Limited | Higher frequency |

## 4. Known Issues

- 单频率不够 → Phase 9.1 多频率编码 (NeRF-style)
