# Phase 6.0: Matrix Multiply — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_01 (前半部分)
> **前置 Phase**: Phase 5.3

## 1. Introduction

Phase 6.0 在 GPU shader 中实现矩阵乘法，是神经网络最基础的构建块。
`output = W * input` (无 bias, 无 activation)。

```
2D input (uv) → [3×2 weight matrix] → 3D output (RGB color)
```

## 2. Core Algorithm

```slang
float3 render(int2 pixel, int2 resolution, Tensor<float, 2> weights) {
    float2 uv = float2(pixel) / float2(resolution);
    // W * uv: 3x2 matmul, 3 outputs = RGB
    return float3(
        weights.getv(int2(0,0)) * uv.x + weights.getv(int2(0,1)) * uv.y,
        weights.getv(int2(1,0)) * uv.x + weights.getv(int2(1,1)) * uv.y,
        weights.getv(int2(2,0)) * uv.x + weights.getv(int2(2,1)) * uv.y
    );
}
```

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_6_0_matmul.slang` | render() with matrix multiply |
| `step_6_0_matmul.py` | Create weight matrix, run shader |

## 4. New Concepts

- GPU Tensor 作为权重矩阵: `Tensor<float, 2>` 存储 3×2 权重
- 逐像素线性变换: 每个像素独立做 matmul
- 从渲染到神经网络: matmul 是 MLP 的第一块积木

## 5. Known Issues

- 无 bias, 无 activation → Phase 6.1 加入 bias + ReLU
- 硬编码 3×2 形状 → Phase 6.2 通用 MLP 层
