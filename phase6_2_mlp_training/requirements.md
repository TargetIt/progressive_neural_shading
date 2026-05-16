# Phase 6.2 Requirements

## New Features (2026-05-16)

- **NetworkParameters**: 参数化的神经网络层 (weights + biases + gradients + Adam)
- **Atomic gradient accumulation**: `InterlockedAdd` for batch training
- **Stochastic sampling**: LCG random pixel selection
- **Full training loop**: 20 steps/frame, loss printed

## Functional Requirements

1. Run `python src/step_6_2_mlp_training.py` to see MLP training
2. 3-panel: Reference | Prediction | Loss
3. Loss decreases during training

## Acceptance Criteria

- [x] Network trains and loss decreases
- [x] Atomic gradient accumulation works
- [x] 3-panel display: Reference / Pred / Loss
- [x] ESC closes window
- [x] Functional equivalence with network step_01
