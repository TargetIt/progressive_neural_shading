# Phase 2.2 Requirements

## New Features (2026-05-16)

- **完整 Mipmap 链**: 多级迭代降采样，每级分辨率减半
- **功能对齐参考**: 等价于 `neural-shading-s25/mipmap/step_02_mipmap`
- **全分辨率窗口**: 1024×1024 输出

## Functional Requirements

1. User can run `python src/step_2_2_mipmap.py` and see mipmap-downsampled BRDF output
2. `downsample()` 支持 steps 参数实现多级降采样链
3. Mipmap 链每级分辨率是上一级的 1/2
4. 输出无明显 artifacts

## Acceptance Criteria

- [x] Window opens without errors at 1024×1024
- [x] Mipmap chain produces correct resolution halving
- [x] No NaN in output
- [x] ESC closes the window
- [x] Functional equivalence with reference step_02_mipmap
