# Phase 1.2: Full BRDF with Textures

## Quick Start

```bash
pip install slangpy
python src/step_1_2_full_brdf.py
```

应该看到 1024×1024 的 PBR 渲染窗口，显示用 Disney BRDF 渲染的石材地面。按 ESC 退出。

## What This Phase Teaches

- 纹理采样: `.getv(pixel)` 从 GPU 纹理读取像素值
- `spy.Tensor.load_from_image()`: JPEG/PNG → GPU Tensor
- MaterialParameters struct: GPU 中组织材质数据的标准模式
- Disney BRDF: 工业级基于物理的渲染模型 (Fresnel, GGX, Smith G, Subsurface)
- ACES 色调映射: HDR 场景 → LDR 显示
- Slang module 系统: `import brdf;` 跨文件代码复用

## New in Phase 1.2

- **brdf.slang**: 完整 Disney BRDF 实现 (118 行)
- **MaterialParameters**: 封装 3 张纹理的采样方法
- **纹理加载**: linearize / scale+offset / grayscale 等参数
- **ACES tonemap**: `tonemap_aces_film()` 在 blit 阶段应用

## Diff from Phase 1.1

| Phase 1.1 | Phase 1.2 |
|-----------|-----------|
| 硬编码蓝色 albedo | 纹理采样 albedo |
| 平面法线 (0,0,1) | 法线贴图 (凹凸) |
| 固定 roughness | 粗糙度纹理 |
| Lambertian + Blinn-Phong | Full Disney BRDF |
| 512×512 | 1024×1024 |
| 无色调映射 | ACES tonemap |

## Using trace.py

```python
from trace import tensor_stats, print_stats, verify_texture_output

# 验证纹理 BRDF 输出是否正常
ok, msg = verify_texture_output(output)
print(f"Texture output check: {msg}")
# => Texture output check: mean=0.2456, std=0.0891

# 如果纹理未加载或渲染失败
ok, msg = verify_texture_output(dark_output)
# => False, "mean=0.0001 (输出太暗或被截断)"
```
