# Phase 2.0 Requirements

## New Features (2026-05-16)

- **Python 手动降采样**: 2×2 box filter 在 NumPy 端实现，每次调用分辨率减半
- **GPU↔CPU 数据传输**: `Tensor.to_numpy()` + `Tensor.from_numpy()`
- **NumPy reshape+mean**: 向量化的 box filter 实现
- **多级降采样**: steps 参数控制降采样级数

## Functional Requirements

1. User can run `python src/step_2_0_downsample.py` and see downsampled BRDF output
2. `downsample_python()` 使用 2×2 box filter 将分辨率减半
3. 降采样后输出仍包含纹理细节 (不是纯色)
4. 复用 Phase 1.2 的 BRDF shader (`step_1_2_full_brdf.slang`)

## Acceptance Criteria

- [x] Window opens without errors
- [x] Output resolution ≤ 原始分辨率的 1/4 (steps=2)
- [x] BRDF rendering + downsampling pipeline works end-to-end
- [x] ESC closes the window
- [x] Phase 1.2 regression verified by running Phase 1.2 tests independently
