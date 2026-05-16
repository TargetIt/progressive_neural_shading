# Phase 1.2 Requirements

## New Features (2026-05-16)

- **纹理加载**: `spy.Tensor.load_from_image()` 从 JPEG 加载 PBR 纹理到 GPU
- **MaterialParameters struct**: 封装 albedo/normal/roughness 纹理的 GPU 数据结构
- **完整 Disney BRDF** (`brdf.slang`): 物理正确的 BRDF 模型
- **ACES 色调映射** (`app.slang`): HDR → LDR 的 filmic tonemapping
- **高分辨率**: 1024×1024 窗口

## Functional Requirements

1. User can run `python src/step_1_2_full_brdf.py` and see PBR-rendered stone pavement
2. 加载 3 张 PBR 纹理: albedo (sRGB→linear), normal ([0,1]→[-1,1]), roughness (grayscale)
3. 编译 `brdf.slang` 和 `step_1_2_full_brdf.slang`
4. 输出非零 RGB 值，像素间有显著变化 (纹理细节可见)
5. ACES 色调映射正确处理 HDR 值

## Acceptance Criteria

- [x] Window opens without errors at 1024×1024
- [x] 3 PBR textures loaded to GPU successfully
- [x] Shader compiles with `import brdf;` dependency
- [x] Output shows visible texture detail (not flat color)
- [x] ACES tonemapping produces viewable result
- [x] ESC closes the window
- [x] Functional equivalence with reference step_01_basicprogram
