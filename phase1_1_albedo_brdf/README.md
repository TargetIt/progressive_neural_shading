# Phase 1.1: Albedo BRDF — Lighting with Hardcoded Color

## Quick Start

```bash
pip install slangpy
python src/step_1_1_albedo_brdf.py
```

应该看到 512×512 的蓝色光照效果 (中心偏亮、边缘偏暗)。按 ESC 退出。

## What This Phase Teaches

- BRDF 的基本概念: 光线如何在材质表面反射
- Lambertian diffuse (漫反射): `diffuse = albedo * max(0, N·L)`
- Blinn-Phong specular (高光): `specular = pow(N·H, 1/roughness)`
- Slang 中 float3 向量的操作: `dot()`, `normalize()`, `pow()`
- Uniform 参数: Python 传入的标量/向量在 GPU 所有线程中共享

## New in Phase 1.1

- **eval_brdf()**: 简化版 BRDF 评估函数 (diffuse + specular)
- **render() 新参数**: `light_dir`, `view_dir` 从 host 端传入
- **硬编码材质**: 蓝色 albedo = `(0.2, 0.4, 1.0)`, 平面法线

## Diff from Phase 1.0

| Phase 1.0 | Phase 1.1 |
|-----------|-----------|
| 纯红色 flat 输出 | 蓝色 BRDF 光照 |
| 无光照计算 | Lambertian + Blinn-Phong |
| 无输入参数 | light_dir + view_dir |
| 24 行 .slang | 62 行 .slang |

## Using trace.py

```python
from trace import tensor_stats, print_stats, verify_lighting_variation

# 验证 BRDF 光照是否产生了像素间变化
ok, msg = verify_lighting_variation(output)
print(f"Lighting check: {msg}")
# => Lighting check: std=0.1234 (有光照变化)

# 如果输出全屏单色，说明光照计算有误
ok, msg = verify_lighting_variation(flat_output, min_std=0.001)
print(f"Check: {msg}")
# => Check: std=0.0000 (无光照变化, 可能全屏单色)
```
