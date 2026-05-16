# Phase 3.0 Requirements

## New Features (2026-05-16)

- **2x 高分辨率渲染**: 在双倍分辨率下渲染 BRDF
- **并排对比显示**: `blit()` offset/size 参数实现左右对比
- **超采样预览**: 2x 渲染 + downsample → 抗锯齿效果

## Functional Requirements

1. Run `python src/step_3_0_hi_res.py` to see side-by-side 1x vs 2x comparison
2. 左半显示原始分辨率 BRDF，右半显示 2x 渲染后降采样的结果
3. Right side should show smoother edges (anti-aliasing effect)

## Acceptance Criteria

- [x] Window opens with side-by-side comparison
- [x] 1x output renders correctly (left half)
- [x] 2x output renders and downsamples correctly (right half)
- [x] ESC closes the window
