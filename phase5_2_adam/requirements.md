# Phase 5.2 Requirements

## New Features (2026-05-16)

- **Adam 优化器**: m/v moments + bias correction + parameter update
- **迭代优化**: 10 steps/frame, loss decreasing
- **optimizer_step**: Slang 实现的 Adam step (float3 和 float 版本)

## Functional Requirements

1. Run `python src/step_5_2_adam.py` to see optimization in action
2. Loss decreases over optimization steps (printed to console)
3. Prediction image gradually approaches reference

## Acceptance Criteria

- [x] Adam optimizer works: loss decreasing
- [x] optimizer_step3 and optimizer_step1 function correctly
- [x] Console shows step number and loss
- [x] ESC closes the window
