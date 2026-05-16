# Phase 3.1 Requirements

## New Features (2026-05-16)

- **4×4 SSAA**: `ssaa_render()` 每输出像素采样 16 个子像素
- **完整 SSAA 管线**: 4x 渲染 → 2 级 downsample → 等价 4×4 SSAA
- **功能对齐参考**: 等价于 `step_03_supersample`

## Functional Requirements

1. Run `python src/step_3_1_ssaa.py` to see left-right SSAA comparison
2. ssaa_render() averages 16 sub-pixel samples per output pixel
3. SSAA side shows visible anti-aliasing improvement

## Acceptance Criteria

- [x] Window opens with side-by-side comparison
- [x] SSAA output has less aliasing than standard
- [x] ESC closes the window
- [x] Functional equivalence with reference step_03_supersample
