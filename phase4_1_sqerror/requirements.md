# Phase 4.1 Requirements

## New Features (2026-05-16)

- **Squared error**: `sq_error()` 计算每通道 (color - reference)^2
- **MSE 输出**: console 打印整体 MSE
- **Light direction 对比**: reference vs prediction 使用不同光源角度

## Functional Requirements

1. Run `python src/step_4_1_sqerror.py` to see 3-panel with MSE
2. MSE is printed to console and non-zero
3. Squared error map shows per-channel RGB error

## Acceptance Criteria

- [x] 3-panel display works (ref / pred / squared error)
- [x] MSE printed to console
- [x] ESC closes the window
