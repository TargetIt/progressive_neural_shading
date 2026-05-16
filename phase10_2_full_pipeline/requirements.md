# Phase 10.2 Requirements

## New Features (2026-05-16)

- **完整神经纹理管线**: Latent → MLP → Material → BRDF → Loss
- **End-to-end 可微渲染**: 从 latent 到 BRDF 的完全可微路径
- **功能对齐**: 等价于 `network/step_05_latent_texture`

## Functional Requirements

1. Run `python src/step_10_2_full_pipeline.py` to see full pipeline
2. Latent texture learns spatial material features
3. BRDF rendering quality improves with training

## Acceptance Criteria

- [x] Full pipeline: latent → MLP → BRDF → loss
- [x] End-to-end gradient flow verified
- [x] Loss decreases, rendering improves
- [x] ESC closes window
- [x] Functional equivalence with network step_05
