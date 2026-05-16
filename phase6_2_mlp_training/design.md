# Phase 6.2: MLP Training — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_01 (完整版)
> **前置 Phase**: Phase 6.1

## 1. Introduction

Phase 6.2 是 Step 6 (Basic Network) 最终形态，等价于参考项目 `network/step_01_basicnetwork`。
完整的单层 MLP 训练: 2 inputs (uv) → 3 outputs (RGB)，训练目标匹配参考图像。

```
NetworkParameters<2,3>: W(3x2) + b(3)
  Input: uv ∈ [0,1]^2
  Output: RGB ∈ [0,1]^3
  Loss: mean((pred - ref)^2)
  Optimizer: Adam with atomic gradient accumulation
```

## 2. Core Architecture

```python
class NetworkParameters(spy.InstanceList):
    weights: Tensor  # 3×2 float
    biases: Tensor   # 3 float
    *_grad: Tensor   # gradient tensors
    m_*, v_*: Tensor # Adam moments

    def optimize(lr, step):  # Adam update
```

### Key Features
- **Atomic gradient accumulation**: `InterlockedAdd` in Slang for batch gradient averaging
- **Stochastic sampling**: `wang_hash(seed)` + LCG for mini-batch
- **20 steps/frame**: 密集训练循环

## 3. Comparison

| Aspect | Phase 6.1 | Phase 6.2 |
|--------|-----------|-----------|
| Structure | Single neuron call | NetworkParameters + Network |
| Weights | Hardcoded | Random init + trained |
| Gradient | None | Atomic accumulation |
| Reference | None | slangstars.png |
| 参考对齐 | 中间状态 | 等价 network step_01 |

## 4. Known Issues

- 单层 MLP 表达能力有限 → Step 7 引入多层网络
- ReLU only → Step 8 探索更多激活函数
