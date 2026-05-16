# Phase 5.3: Full Training Loop

## Quick Start

```bash
python src/step_5_3_train.py
```

完整训练管线: Ref | Pred | Loss。观察 loss 逐渐逼近 baseline。按 ESC 退出。

## What This Phase Teaches

- 完整可微渲染训练管线: differentiable rendering + AD + Adam
- SSAA reference sampling: wang_hash + LCG
- Learning rate decay: 线性衰减 schedule

## New in Phase 5.3

- **calculate_grads**: SSAA-based gradient computation
- **wang_hash**: 随机种子生成
- **init3/init1**: GPU 参数初始化
- **LR decay**: 0.002 → 0.0002

## Diff from Phase 5.2

| Phase 5.2 | Phase 5.3 |
|-----------|-----------|
| 10 steps/frame | 50 steps/frame |
| Simple ref | SSAA ref |
| Constant LR | Linear decay |
| No baseline | Baseline comparison |
