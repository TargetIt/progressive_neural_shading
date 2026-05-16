# Phase 4.2: Loss Visualization — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_04_loss (完整版)
> **前置 Phase**: Phase 4.1

## 1. Introduction

Phase 4.2 是 Step 4 最终形态，体现完整的 Loss Visualization 管线。
核心思想: 将纹理降采样 → 低分辨率渲染 → 与高分辨率 reference 对比 → 可视化 loss。

```
Full-res textures ──→ full-res render ──→ downsample → reference
Low-res textures ──→ render ──────────→ prediction
                            ↓
              loss(reference, prediction) → heatmap
```

这模拟了"神经网络应该从低分辨率输入预测高分辨率输出"的场景。

## 2. Core Algorithm

```slang
float3 loss(int2 pixel, float3 reference, MaterialParameters mat, float3 L, float3 V) {
    float3 color = render(pixel, mat, L, V);
    float3 error = color - reference;
    return error * error;  // squared error per channel
}
```

## 3. Pipeline

```
1. Full-res textures → render → output (H×W)
2. output → downsample 2 steps → reference (H/4 × W/4)
3. textures → downsample 2 steps → low-res textures (H/4 × W/4)
4. low-res textures → render → prediction (H/4 × W/4)
5. loss(reference, prediction) → heatmap
```

## 4. Architecture

| 文件 | 职责 |
|------|------|
| `step_4_2_loss_viz.slang` | MaterialParameters + render() + loss() + downsample3/1 |
| `step_4_2_loss_viz.py` | Full pipeline + 3-panel display |
| `trace.py` | loss_stats() |

## 5. Comparison

| Aspect | Phase 4.1 | Phase 4.2 |
|--------|-----------|-----------|
| Reference | Different light angle | Full-res → downsample |
| Prediction | Same res, different light | Low-res textures → render |
| Pipeline | Simple sq_error | Full loss viz pipeline |
| 参考对齐 | 中间状态 | 等价 step_04_loss |

## 6. Known Issues

- Loss 仅用于可视化，尚未用于优化 → Step 5 引入梯度下降训练
- 纹理降采样使用简单 box filter (非 mipmap) → 对某些材质可能不理想
