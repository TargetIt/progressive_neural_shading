# Phase 5.3 Requirements

## New Features (2026-05-16)

- **完整训练管线**: 可微渲染 + SSAA 参考 + Adam + LR 衰减
- **SSAA 参考采样**: `wang_hash(seed)` + LCG 随机方向
- **Learning rate 衰减**: 线性: 0.002 → 0.0002 (3000 steps)
- **Baseline loss**: 对比优化前后的 loss

## Functional Requirements

1. Run `python src/step_5_3_train.py` to see full training pipeline
2. Loss decreases from baseline over optimization
3. Console shows Step / Loss / Baseline

## Acceptance Criteria

- [x] Full training loop works: loss decreases
- [x] SSAA reference computed correctly
- [x] LR decay active
- [x] ESC closes window
- [x] Functional equivalence with step_05_train
