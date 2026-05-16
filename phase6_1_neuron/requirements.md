# Phase 6.1 Requirements

## New Features (2026-05-16)

- **Bias 向量**: `Tensor<float, 1>` 存储 3 元素 bias
- **ReLU 激活**: `max(0, x)` — 非线性激活函数

## Functional Requirements

1. Run `python src/step_6_1_neuron.py` to see complete neuron output
2. Output has no negative values (ReLU property)
3. Dark regions where W*uv+b < 0 (ReLU kills those)

## Acceptance Criteria

- [x] Neuron: matmul + bias + ReLU works
- [x] No negative output values
- [x] ESC closes the window
