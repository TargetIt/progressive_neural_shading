# Phase 10.1 Requirements

## New Features (2026-05-16)

- **可训练 Latent Texture**: RWTensor + gradient accumulation
- **End-to-end 优化**: latent + MLP 同时训练

## Functional Requirements

1. Run `python src/step_10_1_trainable_latent.py`
2. Latent texture receives gradients and updates
3. Loss decreases: latent learns spatial features

## Acceptance Criteria

- [x] Trainable latent: gradient flow to latent texture
- [x] Loss decreases during training
- [x] ESC closes the window
