# Phase 1.1 Requirements

## New Features (2026-05-16)

- **BRDF 光照模型** (`eval_brdf`): Lambertian diffuse + Blinn-Phong specular
- **Uniform 参数传递**: `light_dir` 和 `view_dir` 从 Python 传入 shader
- **硬编码材质**: 蓝色 albedo `(0.2, 0.4, 1.0)`, roughness 0.3
- **平面法线**: normal = `(0, 0, 1)`, 所有像素统一

## Functional Requirements

1. User can run `python src/step_1_1_albedo_brdf.py` and see blue shaded lighting
2. 输出有明显的亮度变化 (中心高光 + 边缘变暗)
3. ESC key closes the window
4. `light_dir` 和 `view_dir` 通过 uniform 参数传入

## Acceptance Criteria

- [x] Window opens without errors
- [x] Output shows lighting variation (not solid color)
- [x] ESC closes the window
- [x] Shader compiles and renders BRDF-calculated pixels
- [x] Phase 1.0 regression verified by running Phase 1.0 tests independently
