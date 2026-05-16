# Phase 4.0: Per-Pixel Diff

## Quick Start

```bash
python src/step_4_0_diff.py
```

三面板: 左=Reference, 中=Prediction, 右=Error Map (灰度)。按 ESC 退出。

## What This Phase Teaches

- Loss 的基础: 如何量化两个渲染结果的差异
- Per-pixel absolute difference: |pred - ref|
- Error visualization: 将误差编码为灰度图像

## New in Phase 4.0

- **abs_diff()**: 逐像素绝对差计算
- **3-panel 显示**: 1536×512 窗口

## Diff from Phase 3.1

| Phase 3.1 | Phase 4.0 |
|-----------|-----------|
| 1x vs SSAA | Reference vs Prediction |
| SSAA 抗锯齿 | Loss 计算 |
| 2 panels | 3 panels |
