# Phase 5.0 Requirements

## New Features (2026-05-16)

- **数值梯度计算**: Central finite differences on roughness
- **梯度可视化**: dL/d(roughness) 作为灰度热图显示
- **Loss + 梯度统计**: console 输出 MSE 和 |grad|_mean

## Functional Requirements

1. Run `python src/step_5_0_numgrad.py` to see gradient heatmap
2. Gradient correctly points toward higher loss when perturbing roughness
3. Console shows MSE and mean absolute gradient

## Acceptance Criteria

- [x] Numerical gradient computed correctly (L(x+h) - L(x-h))/(2h)
- [x] Gradient heatmap displayed alongside reference
- [x] Console output shows MSE and gradient magnitude
- [x] ESC closes the window
