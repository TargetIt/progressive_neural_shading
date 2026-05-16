# Phase 4.2: Loss Visualization

## Quick Start

```bash
python src/step_4_2_loss_viz.py
```

三面板: Ref (左) | Loss Heatmap (中) | Pred (右)。按 ESC 退出。

## What This Phase Teaches

- Loss visualization: 如何使用误差热图理解渲染质量
- Low-res → high-res 问题: 神经网络的目标设定
- Full pipeline: texture downsample → render → loss

## New in Phase 4.2

- **loss()**: squared error between reference and prediction
- **完整管线**: texture downsample + render + loss
- **大窗口**: 3092×1024 三面板

## Diff from Phase 4.1

| Phase 4.1 | Phase 4.2 |
|-----------|-----------|
| 不同光源对比 | Low-res vs High-res |
| 单级渲染 | Full down/up pipeline |
| MSE in console | Loss heatmap |
