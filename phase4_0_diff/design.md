# Phase 4.0: Per-Pixel Diff — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_04_loss (前半部分)
> **前置 Phase**: Phase 3.1

## 1. Introduction

Phase 4.0 计算两个 BRDF 渲染之间的逐像素绝对差 (Per-Pixel Absolute Difference)。
三面板显示: Reference (左) | Prediction (中) | Error Map (右)。

```
ref = BRDF(original roughness)
pred = BRDF(modified roughness)  ← 故意制造差异
diff = |pred - ref| → grayscale error map
```

## 2. Core Algorithm

```slang
float3 abs_diff(int2 pixel, float3 reference, MaterialParameters mat, float3 L, float3 V) {
    float3 color = render(pixel, mat, L, V);
    float3 err = abs(color - reference);
    float gray = (err.x + err.y + err.z) / 3.0;
    return float3(gray);
}
```

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_4_0_diff.slang` | MaterialParameters + render() + abs_diff() |
| `step_4_0_diff.py` | 3-panel display: ref / pred / diff |
| `trace.py` | Tensor stats + error_stats() |

## 4. Comparison

| Aspect | Phase 3.1 | Phase 4.0 |
|--------|-----------|-----------|
| 渲染 | 1x vs SSAA | reference vs prediction |
| 显示 | 2 panels | 3 panels |
| 新增 | ssaa_render | abs_diff + error map |

## 5. Known Issues

- 使用 absolute difference 而非 squared error → Phase 4.1 引入 squared error
- Error 是 per-pixel 的，不是单一标量 loss → Phase 4.2 可视化 loss landscape
