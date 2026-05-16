# Phase 6.2: MLP Training

## Quick Start

```bash
python src/step_6_2_mlp_training.py
```

单层 MLP 训练匹配 slangstars.png。3-panel: Ref | Pred | Loss。按 ESC 退出。

## What This Phase Teaches

- NetworkParameters: 参数化网络层的 Python/Slang 绑定
- Atomic gradient accumulation: GPU 并行梯度求和
- Mini-batch stochastic training: LCG + wang_hash

## New in Phase 6.2

- **NetworkParameters<2,3>**: weights + biases + gradients + Adam moments
- **Atomic accumulation**: `InterlockedAdd` for multi-thread gradient
- **Batch training**: 64×64 mini-batch per step

## Diff from Phase 6.1

| Phase 6.1 | Phase 6.2 |
|-----------|-----------|
| Hardcoded W, b | Learned W, b |
| No gradient | Atomic grad accumulation |
| Single forward | Forward + backward + Adam |
