# Phase 4.0 Requirements

## New Features (2026-05-16)

- **Per-pixel absolute difference**: `abs_diff()` 计算两个渲染输出的逐像素 |pred - ref|
- **三面板显示**: Reference / Prediction / Error Map
- **Error 可视化**: 灰度错误图 (亮=误差大, 暗=误差小)

## Functional Requirements

1. Run `python src/step_4_0_diff.py` to see 3-panel comparison
2. Left: reference (original roughness)
3. Center: prediction (modified roughness)
4. Right: grayscale abs error map

## Acceptance Criteria

- [x] 3-panel display works
- [x] Error map shows brighter pixels where roughness differs
- [x] ESC closes the window
