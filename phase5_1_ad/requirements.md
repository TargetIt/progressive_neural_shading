# Phase 5.1 Requirements

## New Features (2026-05-16)

- **Slang 自动微分**: `[Differentiable]` + `[BackwardDerivativeOf]` + `bwd_diff`
- **RWTensor**: 可读写 GPU Tensor 用于存储梯度
- **多参数梯度**: 同时计算 albedo/normal/roughness 的梯度

## Functional Requirements

1. Run `python src/step_5_1_ad.py` to see gradient computation
2. `bwd_diff(loss)` computes gradients for all trainable params
3. Console shows per-parameter gradient magnitudes

## Acceptance Criteria

- [x] AD gradients computed correctly (no NaN)
- [x] Gradient tensors properly zeroed each frame
- [x] ESC closes the window
