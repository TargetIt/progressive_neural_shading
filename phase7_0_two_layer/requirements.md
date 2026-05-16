# Phase 7.0 Requirements

## New Features (2026-05-16)

- **双层网络**: 2→16 (ReLU) → 3 (linear)
- **多 NetworkParameters**: layer0 + layer1 组成 Network
- **隐藏层**: 16 个隐藏神经元, 比单层有更强的表达能力

## Functional Requirements

1. Run `python src/step_7_0_two_layer.py` to see two-layer network output
2. Layer0: 2→16 with ReLU; Layer1: 16→3 linear

## Acceptance Criteria

- [x] Two-layer forward pass works
- [x] Output shows non-trivial pattern (not flat)
- [x] ESC closes the window
