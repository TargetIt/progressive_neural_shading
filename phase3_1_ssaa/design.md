# Phase 3.1: SSAA Pipeline — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_03_supersample (完整版)
> **前置 Phase**: Phase 3.0

## 1. Introduction

Phase 3.1 是 Step 3 最终形态，实现 4×4 SSAA (Supersampling Anti-Aliasing)。
新增 `ssaa_render()` 函数，对每个输出像素采样 4×4=16 个子像素并求平均。

```
render 4x resolution → downsample 4x = SSAA
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ PNG纹理   │  →   │ 4x 渲染  │  →   │ 2×2 降采样│
  │ (1K×1K)  │      │ (4K×4K)  │      │ 2 次     │
  └──────────┘      └──────────┘      └──────────┘
                        ↓                  ↓
                  ssaa_render()      downsample3()
                  每像素 16 采样       box filter
```

## 2. Core Algorithm

```slang
float3 ssaa_render(int2 pixel, MaterialParameters mat, float3 L, float3 V) {
    int2 base = pixel * 4;
    float3 sum = 0.0;
    for (int dx = 0; dx < 4; dx++)
        for (int dy = 0; dy < 4; dy++)
            sum += render(base + int2(dx, dy), mat, L, V);
    return sum / 16.0;
}
```

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_3_1_ssaa.slang` | MaterialParameters + render() + ssaa_render() + downsample3/1 |
| `step_3_1_ssaa.py` | 1x vs SSAA 渲染 + 并排显示 |
| `trace.py` | Tensor 统计 + SSAA 对比 |

## 4. Comparison

| Aspect | Phase 3.0 | Phase 3.1 |
|--------|-----------|-----------|
| 超采样 | 2x | 4×4 = 16x |
| 子像素方法 | 无 (只有一个 2x render) | ssaa_render() 循环 16 采样 |
| 画质 | 轻微改善 | 显著抗锯齿 |
| 参考对齐 | 中间状态 | 等价 step_03_supersample |

## 5. Known Issues

- 循环内调用 render() 无法自动微分 → Phase 5 用 `[Differentiable]` 解决
- 性能: 每像素 16 次 BRDF 评估 → 后续 Phase 将引入神经网络加速
