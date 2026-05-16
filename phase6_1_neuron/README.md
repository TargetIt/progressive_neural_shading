# Phase 6.1: Neuron Complete

## Quick Start

```bash
python src/step_6_1_neuron.py
```

完整神经元: W * uv + b → ReLU。暗区 = ReLU 输出 0。按 ESC 退出。

## What This Phase Teaches

- 完整神经元: linear(W*uv + b) → ReLU activation
- Bias 的作用: 平移输出
- ReLU 的作用: 引入非线性, 稀疏激活

## New in Phase 6.1

- **Bias**: 3-element float bias vector
- **ReLU**: max(0, x) activation

## Diff from Phase 6.0

| Phase 6.0 | Phase 6.1 |
|-----------|-----------|
| W * uv | W * uv + b |
| Linear | ReLU(W*uv + b) |
| No dead zones | Dark (dead) regions |
