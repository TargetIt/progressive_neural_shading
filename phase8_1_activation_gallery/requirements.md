# Phase 8.1 Requirements

## New Features (2026-05-16)

- **Activation Gallery**: 并排对比 5+ 种激活函数
- **LeakyReLU, GELU, Swish**: 比 ReLU 更优的激活函数
- **功能对齐**: 等价于 `network/step_03_better_activations`

## Functional Requirements

1. Run `python src/step_8_1_activation_gallery.py` to compare activations
2. Side-by-side display of multiple activation outputs
3. Dead neuron percentages reported for each

## Acceptance Criteria

- [x] All activation functions displayed
- [x] Comparative stats shown (dead %, active mean)
- [x] ESC closes window
- [x] Functional equivalence with network step_03
