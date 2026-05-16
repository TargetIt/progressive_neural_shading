# Phase 10.1: Trainable Latent Texture

## Quick Start

```bash
python src/step_10_1_trainable_latent.py
```

可训练的 latent texture: 通过反向传播学习空间特征。按 ESC 退出。

## What This Phase Teaches

- Trainable latent: 从固定 → 可学习的转变
- RWTensor + BackwardDerivativeOf: gradient to latent
- End-to-end optimization: latent + MLP 同时优化

## New in Phase 10.1

- **Trainable latent**: RWTensor with grad accumulation
- **latent optimizer_step**: Adam step for latent texture

## Diff from Phase 10.0

| Phase 10.0 | Phase 10.1 |
|-----------|-----------|
| Fixed latent | Trainable latent |
| No latent grad | Latent gradient flow |
