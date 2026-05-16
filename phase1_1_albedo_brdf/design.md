# Phase 1.1: Albedo BRDF — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_01 前半部分 (简化版)
> **前置 Phase**: Phase 1.0

## 1. Introduction / 架构概览

Phase 1.1 在 Phase 1.0 的 GPU 调用骨架之上，引入了 **BRDF 光照模型**。
使用硬编码的蓝色 albedo 替代纹理采样，让学习者聚焦于 BRDF 公式本身。

```
┌──────────────────────────────────────────────────┐
│                step_1_1_albedo_brdf.py             │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │ App 框架 │ → │ Slang 编译 │ → │ GPU 执行  │  │
│  │ app.py   │   │ .slang→GPU │   │ 每像素并行 │  │
│  └──────────┘   └────────────┘   └───────────┘  │
│       │               │                │         │
│       v               v                v         │
│  窗口+设备     step_1_1.slang     BRDF 蓝色光照  │
│  + light_dir   eval_brdf()       蓝色球面效果    │
│  + view_dir    Lambert+Specular                  │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

Phase 1.0 只返回纯红色，没有任何光照计算。真实渲染需要 BRDF:
- **为什么要 BRDF**: 描述光线如何在材质表面反射
- **为什么先硬编码颜色**: 让学习者先理解 BRDF 公式，避免纹理采样的干扰
- **为什么用简化 BRDF**: Disney BRDF 有 100+ 行，初学者难以消化；先学 Lambertian + Blinn-Phong

## 3. Algorithm and Theory / 核心算法

### 3.1 Lambertian Diffuse (漫反射)

```
NdotL = max(0, dot(normal, light_dir))
diffuse = albedo * NdotL
```

- `N·L > 0`: 光线照到正面 → 有漫反射
- `N·L = 0`: 光线切线方向 → 漫反射为零
- `N·L < 0`: 光线在背面 → clamp 到 0

### 3.2 Blinn-Phong Specular (高光)

```
half_vec = normalize(light_dir + view_dir)
NdotH = max(0, dot(normal, half_vec))
specular = pow(NdotH, 1.0 / roughness)
```

### 3.3 组合

```
return (diffuse + specular * 0.5) * light_intensity
```

## 4. Architecture / 架构

### 4.1 Module Breakdown

| 文件 | 职责 | 行数 |
|------|------|------|
| `app.py` | 窗口创建, GPU 设备, blit 到屏幕 | ~94 |
| `app.slang` | blit helper (Tensor→屏幕) | ~22 |
| `step_1_1_albedo_brdf.slang` | BRDF 着色器: eval_brdf() + render() | ~62 |
| `step_1_1_albedo_brdf.py` | 入口: 加载 shader, 设置光照参数 | ~34 |
| `trace.py` | Tensor 统计 + 光照变化验证 | ~40 |

### 4.2 新增 API

```python
# 传入 uniform 参数 (所有像素共享)
module.render(
    pixel=spy.call_id(),
    light_dir=spy.float3(0.3, 0.2, 1.0),  # 光源方向
    view_dir=spy.float3(0.0, 0.0, 1.0),    # 观察方向
    _result=output,
)
```

### 4.3 新增 Slang 函数

```slang
float3 eval_brdf(albedo, light_dir, view_dir, normal, roughness)
float3 render(int2 pixel, float3 light_dir, float3 view_dir)
```

## 5. Processing Flow / 执行流程

```
1. App.__init__()
   ├── spy.Window(512, 512)
   ├── spy.create_device()
   └── spy.Module.load(app.slang)

2. step_1_1_albedo_brdf.py
   ├── spy.Module.load(step_1_1_albedo_brdf.slang)
   ├── spy.Tensor.empty(512, 512, float3)
   └── light_dir = (0.3, 0.2, 1.0), view_dir = (0, 0, 1)

3. while app.process_events():
   ├── module.render(pixel=...)
   │   └── GPU: 对每个像素并行执行 render()
   │       ├── albedo = (0.2, 0.4, 1.0)      // 硬编码蓝色
   │       ├── roughness = 0.3
   │       ├── normal = (0, 0, 1)              // 平面法线
   │       ├── eval_brdf(...)
   │       │   ├── diffuse = albedo * max(0, N·L)
   │       │   ├── specular = pow(max(0, N·H), 1/roughness)
   │       │   └── return diffuse + specular * 0.5
   │       └── return brdf * light_intensity
   ├── app.blit(output)
   └── app.present()
```

## 6. Comparison / 对比

| Aspect | Phase 1.0 | Phase 1.1 | Change |
|--------|-----------|-----------|--------|
| 着色器输出 | 纯红色 `float3(1,0,0)` | BRDF 光照结果 | +BRDF 计算 |
| 输入参数 | 仅 `pixel` | `pixel` + `light_dir` + `view_dir` | +2 uniform |
| Albedo | 无 | 硬编码蓝色 (0.2, 0.4, 1.0) | 新增 |
| 光照模型 | 无 | Lambertian + Blinn-Phong | 新增 |
| eval_brdf() | ❌ | ✅ (简化版) | 新增 |
| 纹理 | 无 | 无 | — |
| 代码行数 (.slang) | 24 | 62 | +38 |

## 7. Known Issues / 遗留问题

- Albedo 是硬编码的，无法显示不同材质的颜色变化
- Normal 统一指向相机 `(0,0,1)`，没有凹凸感
- Roughness 是固定值 0.3
- 下一 Phase (1.2) 将引入纹理采样和完整的 MaterialParameters struct
