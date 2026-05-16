# Phase 5.0: Numerical Gradient

## Quick Start

```bash
python src/step_5_0_numgrad.py
```

左: Reference render。右: 数值梯度热图 (dL/dRoughness)。按 ESC 退出。

## What This Phase Teaches

- 数值梯度: Central finite differences dL/dx ≈ (L(x+h)-L(x-h))/(2h)
- 为什么需要梯度: 优化/训练的数学基础
- 数值梯度的局限: 慢 (2 次额外前向 pass)、精度依赖 h

## New in Phase 5.0

- **数值梯度计算**: roughness ± h → loss 差分
- **梯度可视化**: 灰度热图

## Diff from Phase 4.2

| Phase 4.2 | Phase 5.0 |
|-----------|-----------|
| Loss only | Loss + Gradient |
| 静态可视化 | 数值优化方向 |
| 2 renders | 3 renders |
