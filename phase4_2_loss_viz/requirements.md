# Phase 4.2 Requirements

## New Features (2026-05-16)

- **完整 Loss 管线**: full-res ref vs low-res pred → squared error heatmap
- **纹理降采样 + 渲染**: 低分辨率材质模拟神经网络输入
- **功能对齐参考**: 等价于 `step_04_loss`

## Functional Requirements

1. Run `python src/step_4_2_loss_viz.py` to see 3-panel loss visualization
2. Left: reference (full-res → downsample)
3. Center: loss heatmap (squared error, no tonemap)
4. Right: prediction (low-res textures → render)

## Acceptance Criteria

- [x] Full loss pipeline works end-to-end
- [x] Loss heatmap shows non-zero error where low-res lacks detail
- [x] ESC closes the window
- [x] Functional equivalence with reference step_04_loss
