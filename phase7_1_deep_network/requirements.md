# Phase 7.1 Requirements

## New Features (2026-05-16)

- **3 层深度网络**: 2→32→32→3 with ReLU
- **完整训练**: Adam + atomic gradients + LCG
- **功能对齐**: 等价于 `network/step_02_multiple_layers`

## Functional Requirements

1. Run `python src/step_7_1_deep_network.py` to see deep network training
2. 3-panel: Reference | Prediction | Loss
3. Loss decreases; prediction approaches reference

## Acceptance Criteria

- [x] 3-layer network trains successfully
- [x] Loss decreases over time
- [x] ESC closes window
- [x] Functional equivalence with network step_02
