# Phase 1.2: Full BRDF with Textures — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_01_basicprogram (完整版)
> **前置 Phase**: Phase 1.1

## 1. Introduction / 架构概览

Phase 1.2 是 Step 1 的最终形态，功能等价于参考项目 `neural-shading-s25/mipmap/step_01_basicprogram`。
在 Phase 1.1 的简化 BRDF 之上，引入**纹理采样**和**完整的 Disney BRDF 模型**。

```
┌──────────────────────────────────────────────────┐
│                step_1_2_full_brdf.py               │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │ 纹理加载 │ → │ Slang 编译 │ → │ GPU 执行  │  │
│  │ JPEG→GPU │   │ .slang→GPU │   │ 每像素并行 │  │
│  └──────────┘   └────────────┘   └───────────┘  │
│       │               │                │         │
│       v               v                v         │
│  3 张 PBR 纹理   step_1_2.slang   PBR 石材渲染  │
│  albedo+normal   import brdf;      Disney BRDF   │
│  +roughness      MaterialParams    ACES tonemap  │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

Phase 1.1 的硬编码颜色无法展示不同材质的视觉效果:
- **纹理采样**: 每像素从 2D 纹理读取不同的 albedo/normal/roughness
- **MaterialParameters struct**: GPU 编程中组织相关数据的标准模式
- **Disney BRDF**: 工业级基于物理的 BRDF 模型，支持 metallic/specular 参数

## 3. Algorithm and Theory / 核心算法

### 3.1 纹理采样

```slang
struct MaterialParameters {
    Tensor<float3, 2> albedo;    // RGB 纹理
    Tensor<float3, 2> normal;    // 法线贴图
    Tensor<float, 2> roughness;  // 粗糙度 (单通道)

    float3 get_albedo(int2 pixel)   { return albedo.getv(pixel); }
    float3 get_normal(int2 pixel)   { return normalize(normal.getv(pixel)); }
    float get_roughness(int2 pixel) { return roughness.getv(pixel); }
};
```

### 3.2 Disney BRDF (brdf.slang)

完整的基于物理的 BRDF 模型:
- **Diffuse**: Schlick Fresnel 驱动的漫反射
- **Specular**: GGX (GTR2) 法线分布 + Smith G 几何项
- **Subsurface**: Hanrahan-Krueger 次表面散射近似
- `[Differentiable]` 标注: 为后续 Phase 的自动微分做准备

### 3.3 ACES 色调映射 (app.slang)

HDR → LDR 显示:
```
tonemap_aces_film(x) = saturate((x*(2.51*x+0.03)) / (x*(2.43*x+0.59)+0.14))
```

## 4. Architecture / 架构

### 4.1 Module Breakdown

| 文件 | 职责 | 行数 |
|------|------|------|
| `app.py` | 窗口创建, GPU 设备, blit + tonemap | ~53 |
| `app.slang` | blit helper + ACES 色调映射 | ~30 |
| `brdf.slang` | 完整 Disney BRDF (DisneyBRDF + eval_brdf) | ~118 |
| `step_1_2_full_brdf.slang` | MaterialParameters struct + render() | ~67 |
| `step_1_2_full_brdf.py` | 纹理加载, 渲染循环 | ~66 |
| `trace.py` | Tensor 统计 + 纹理输出验证 | ~40 |

### 4.2 Key APIs (新增)

```python
# 从 JPEG/PNG 加载纹理到 GPU
albedo_map = spy.Tensor.load_from_image(device, "albedo.jpg", linearize=True)
normal_map = spy.Tensor.load_from_image(device, "normal.jpg", scale=2, offset=-1)
roughness_map = spy.Tensor.load_from_image(device, "roughness.jpg", grayscale=True)

# 传入 struct 参数 (嵌套字典)
module.render(
    pixel=spy.call_id(),
    material={"albedo": albedo, "normal": normal, "roughness": roughness},
    ...
)
```

## 5. Processing Flow / 执行流程

```
1. 纹理加载
   ├── albedo_map: JPEG → GPU Tensor (sRGB → linear)
   ├── normal_map: JPEG → GPU Tensor ([0,1] → [-1,1])
   └── roughness_map: JPEG → GPU Tensor (grayscale)

2. 每帧循环
   ├── module.render(pixel=..., material=...)
   │   └── GPU: 对每个像素并行
   │       ├── albedo = material.get_albedo(pixel)
   │       ├── normal = material.get_normal(pixel)
   │       ├── roughness = material.get_roughness(pixel)
   │       ├── eval_brdf(albedo, light_dir, view_dir, normal, roughness, 0, 1)
   │       │   └── DisneyBRDF(...)  // 完整 PBR 管线
   │       └── return brdf * light_intensity
   ├── app.blit(output, tonemap=True)
   │   └── GPU: ACES 色调映射 (HDR → LDR)
   └── app.present()
```

## 6. Comparison / 对比

| Aspect | Phase 1.1 | Phase 1.2 | Change |
|--------|-----------|-----------|--------|
| Albedo | 硬编码 `(0.2,0.4,1.0)` | 纹理采样 `albedo.getv(pixel)` | 逐像素变化 |
| Normal | 固定 `(0,0,1)` | 法线贴图采样 | 凹凸细节 |
| Roughness | 固定 `0.3` | 粗糙度纹理采样 | 逐像素变化 |
| BRDF | Lambertian + Blinn-Phong | 完整 Disney BRDF | 物理正确 |
| 色调映射 | 无 (clamp) | ACES filmic | HDR 显示 |
| 文件数 | 5 | 7 (+brdf.slang +assets/) | +2 |
| 纹理依赖 | 无 | 3 张 PBR 纹理 | 新增 |
| Window | 512×512 | 1024×1024 | 高分辨率 |

## 7. Known Issues / 遗留问题

- 使用全分辨率纹理渲染，大窗口下性能可能不足 → 下一 Step 引入 Mipmap
- Disney BRDF 是全分辨率评估的，无 LOD → Phase 2 解决
- 单采样 (1 sample/pixel) → Phase 3 引入超采样抗锯齿
