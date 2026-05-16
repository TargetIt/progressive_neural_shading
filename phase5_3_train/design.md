# Phase 5.3: Full Training Loop — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_05_train (完整版)
> **前置 Phase**: Phase 5.2

## 1. Introduction

Phase 5.3 是 Step 5 (Training) 最终形态，等价于参考项目 `step_05_train`。
完整的可微渲染训练管线: 可微 BRDF → AD 梯度 → Adam 优化 → LR 衰减。

```
Full-res textures → render → downsample → reference
                        ↓
Trainable params → render → prediction → loss(SSAA ref, pred)
                        ↓
              bwd_diff → gradient → Adam step → param update
```

## 2. Key Features

- **SSAA 参考**: `wang_hash(seed)` + LCG 随机方向, 4× 子像素
- **50 steps/frame**: 密集优化循环
- **Learning rate decay**: 线性从 0.002 衰减到 0.0002
- **Baseline comparison**: 显示优化前 loss vs 优化后 loss

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_5_3_train.slang` | MaterialParameters (AD) + render + loss + calculate_grads + init + optimizer_step + downsample |
| `step_5_3_train.py` | Full training loop: 50 steps/frame, LR decay |
| `trace.py` | training_progress() |

## 4. Comparison

| Aspect | Phase 5.2 | Phase 5.3 |
|--------|-----------|-----------|
| Steps/frame | 10 | 50 |
| Reference | Simple render | SSAA with wang_hash |
| LR | Constant | Linear decay |
| Baseline | None | Original loss comparison |
| 参考对齐 | 中间状态 | 等价 step_05_train |

## 5. Known Issues

- 训练 3000 步后收敛速度放缓 → 下一 Step 引入神经网络加速
- 纹理参数 vs 神经网络权重: 这是 linear texture optimization, 不是 MLP
