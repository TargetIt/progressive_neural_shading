# Phase 9.0 Requirements

## New Features (2026-05-16)

- **Sin 位置编码**: 将 UV 坐标通过 sin() 映射到高维
- **Spectral bias 缓解**: 帮助 MLP 学习高频纹理细节

## Functional Requirements

1. Run `python src/step_9_0_sin_encoding.py` to see sin-encoded network output
2. Input is sin(freq * UV) instead of raw UV
3. Output shows improved high-frequency detail

## Acceptance Criteria

- [x] Sin encoding applied to UV input
- [x] Network trains with encoded input
- [x] ESC closes the window
