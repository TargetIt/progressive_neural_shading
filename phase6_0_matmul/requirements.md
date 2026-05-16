# Phase 6.0 Requirements

## New Features (2026-05-16)

- **GPU 矩阵乘法**: 每个像素独立计算 W * input (3×2 matmul)
- **权重 Tensor**: `Tensor<float, 2>` 存储可学习的权重矩阵
- **UV 坐标输入**: 归一化像素坐标作为网络输入

## Functional Requirements

1. Run `python src/step_6_0_matmul.py` to see matmul color gradient
2. R channel varies with u, G with v, B with both

## Acceptance Criteria

- [x] Matmul output shows color gradient (not flat)
- [x] Weights correctly map uv → RGB
- [x] ESC closes the window
