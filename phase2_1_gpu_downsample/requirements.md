# Phase 2.1 Requirements

## New Features (2026-05-16)

- **GPU 并行降采样**: `downsample3()` 和 `downsample1()` 在 Slang 中实现 2×2 box filter
- **自包含 shader**: 单个 .slang 文件同时包含 render() 和 downsample() 函数
- **零 GPU↔CPU 传输**: 降采样全程在 GPU 端完成
- **多级循环降采样**: Python 端 for 循环驱动多次 GPU 降采样

## Functional Requirements

1. User can run `python src/step_2_1_gpu_downsample.py` and see GPU-downsampled BRDF output
2. `downsample3()` 在 GPU 上对 float3 Tensor 做 2×2 box filter
3. `downsample1()` 在 GPU 上对 float Tensor 做 2×2 box filter
4. 降采样输出无 NaN
5. 性能优于 Phase 2.0 (无 GPU↔CPU 传输)

## Acceptance Criteria

- [x] Window opens without errors
- [x] GPU downsample produces correct resolution (steps=2 → /4)
- [x] Output has no NaN artifacts
- [x] ESC closes the window
- [x] Phase 2.0 regression verified by running Phase 2.0 tests independently
