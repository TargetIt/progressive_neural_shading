# Phase 3.0: Higher Resolution Render — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_03_supersample (前半部分)
> **前置 Phase**: Phase 2.2

## 1. Introduction / 架构概览

Phase 3.0 演示高分辨率渲染 + 降采样的效果。在同一帧中渲染 1x 和 2x 分辨率，并排对比。

```
┌──────────────────────────────────────────────────┐
│                step_3_0_hi_res.py                  │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐   │
│  │ 1x 渲染  │   │ 2x 渲染  │   │ 并排显示  │   │
│  │ render() │   │ render() │   │ blit ×2  │   │
│  └──────────┘   └──────────┘   └───────────┘   │
│       │              │               │          │
│       v              v               v          │
│  H×W BRDF    2H×2W→H×W        左: 1x 原始      │
│               downsample       右: 2x→down      │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

- **理解超采样**: 高分辨率渲染 + 降采样 = 最简单的抗锯齿
- **并排对比**: 直观看到 1x 和 2x 渲染后降采样的画质差异
- **中间步骤**: 为 Phase 3.1 的完整 SSAA 做铺垫

## 3. Algorithm and Theory / 核心算法

### 超采样原理

```
2x 渲染: 每个源像素对应 2×2 个输出像素
降采样: 4 个像素 box filter 平均 → 1 个像素
效果: 减少 aliasing (锯齿)
```

### 并排显示

```python
app.blit(output_1x, size=(512, 1024), offset=(0, 0))    # 左半
app.blit(output_2x_down, size=(512, 1024), offset=(512, 0))  # 右半
```

## 4. Architecture

| 文件 | 职责 |
|------|------|
| `step_3_0_hi_res.slang` | render() + downsample3/1 + MaterialParameters |
| `step_3_0_hi_res.py` | 1x/2x rendering + side-by-side blit |
| `trace.py` | Tensor 统计 + 1x vs 2x 对比 |

## 5. Comparison

| Aspect | Phase 2.2 | Phase 3.0 |
|--------|-----------|-----------|
| 渲染次数 | 1 | 2 (1x + 2x) |
| 输出 | 单画面 | 左右并排对比 |
| 超采样 | 无 | 2x 渲染 + 降采样 |
| 画质 | baseline | 可见抗锯齿效果 |

## 6. Known Issues

- 手动指定 offset/size 不够灵活 → Phase 3.1 用 SSAA 自动管理
- 只有 stateless 的 2x 超采样，非用户可选
