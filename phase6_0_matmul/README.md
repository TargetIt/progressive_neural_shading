# Phase 6.0: Matrix Multiply

## Quick Start

```bash
python src/step_6_0_matmul.py
```

看到按 uv 坐标线性渐变的三色输出 (matmul 结果)。按 ESC 退出。

## What This Phase Teaches

- GPU 上的矩阵乘法: W * input (线性层)
- Tensor 作为权重: 3×2 矩阵 → RGB 输出
- 神经网络的基础构建块

## New in Phase 6.0

- **Matrix Multiply**: W * uv (no bias, no activation)
- **Weight Tensor**: 3×2 float 矩阵

## Context

这是 neural network step_01 的第一步。Next: Phase 6.1 → Neuron (matmul + bias + activation)
