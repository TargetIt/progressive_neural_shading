# Phase 6.1: Neuron Complete — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_01
> **前置 Phase**: Phase 6.0

## 1. Introduction

Phase 6.1 在 Phase 6.0 的 matmul 基础上添加 bias 和 ReLU activation，
构成一个完整的神经元: `output = max(0, W * input + b)`。

```
Phase 6.0: output = W * uv           (线性变换)
Phase 6.1: output = max(0, W*uv + b) (线性 + bias + ReLU)
```

## 2. Core Algorithm

```slang
float3 render(int2 pixel, int2 resolution, Tensor<float,2> weights, Tensor<float,1> biases) {
    float2 uv = float2(pixel) / float2(resolution);
    float3 linear = /* matmul W * uv */
        float3(weights[0,0]*uv.x + weights[0,1]*uv.y + biases[0],
               weights[1,0]*uv.x + weights[1,1]*uv.y + biases[1],
               weights[2,0]*uv.x + weights[2,1]*uv.y + biases[2]);
    return max(float3(0), linear);  // ReLU
}
```

## 3. New Concepts

- **Bias**: 偏置项让神经元即使输入为 0 也有输出
- **ReLU**: max(0, x) — 引入非线性, 关闭负值神经元

## 4. Comparison

| Aspect | Phase 6.0 | Phase 6.1 |
|--------|-----------|-----------|
| Operation | W * uv | W * uv + b |
| Activation | None | ReLU |
| 非线性性 | Linear | Non-linear |

## 5. Known Issues

- 单个神经元不够 → Phase 6.2 构建 MLP (多层+多神经元)
