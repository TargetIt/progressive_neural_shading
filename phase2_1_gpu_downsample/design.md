# Phase 2.1: GPU Downsample — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_02_mipmap (GPU 并行化)
> **前置 Phase**: Phase 2.0

## 1. Introduction / 架构概览

Phase 2.1 将 2×2 box filter 降采样从 Python/NumPy 迁移到 GPU。
所有计算在 GPU 端完成，消除了 GPU↔CPU 的数据传输瓶颈。

```
┌──────────────────────────────────────────────────┐
│              step_2_1_gpu_downsample.py            │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │ GPU 渲染 │ → │ GPU 降采样 │ → │ 显示      │  │
│  │ render() │   │downsample()│   │ blit()    │  │
│  └──────────┘   └────────────┘   └───────────┘  │
│       │               │                │         │
│       v               v                v         │
│  Disney BRDF    2x2 box filter    分辨率/4 输出  │
│  import brdf    全部在 GPU 并行    零 CPU 参与   │
│                  vs Phase 2.0                     │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

Phase 2.0 的 Python 降采样有严重的性能问题: `to_numpy()` 和 `from_numpy()` 要走 PCIe 总线。
- **GPU 降采样的优势**: 像素间完全并行，无 CPU 参与，速度提升 100-1000x
- **为什么分 downsample3/downsample1**: float3 通道和单通道的 Tensor 类型不同
- **为什么合并 render+downsample**: 减少 shader 切换，理解多功能 shader 模块

## 3. Algorithm and Theory / 核心算法

### 3.1 GPU 并行 Box Filter

```slang
float3 downsample3(int2 pixel, Tensor<float3, 2> source)
{
    float3 res = 0;
    res += source.getv(pixel * 2 + int2(0, 0));
    res += source.getv(pixel * 2 + int2(1, 0));
    res += source.getv(pixel * 2 + int2(0, 1));
    res += source.getv(pixel * 2 + int2(1, 1));
    return res * 0.25;  // 平均值
}
```

每个输出像素独立并行计算其对应的 4 个输入像素的平均值。`pixel * 2` 从输出坐标映射到输入坐标。

### 3.2 与 Phase 2.0 对比

```
Phase 2.0: GPU → to_numpy() → CPU(reshape+mean) → from_numpy() → GPU
Phase 2.1: GPU → render() → downsample() → blit()  (全部在 GPU)
```

## 4. Architecture / 架构

### 4.1 Module Breakdown

| 文件 | 职责 | 行数 |
|------|------|------|
| `app.py` | 窗口创建, GPU 设备, blit + tonemap | ~53 |
| `app.slang` | blit helper + ACES 色调映射 | ~30 |
| `brdf.slang` | 完整 Disney BRDF | ~118 |
| `step_2_1_gpu_downsample.slang` | render() + downsample3/1 | ~55 |
| `step_2_1_gpu_downsample.py` | 入口: 渲染 + GPU 降采样循环 | ~44 |
| `trace.py` | Tensor 统计 + GPU 降采样验证 | ~45 |

### 4.2 Shader 函数

```slang
float3 render(int2 pixel, MaterialParameters mat, float3 light_dir, float3 view_dir)
float3 downsample3(int2 pixel, Tensor<float3, 2> source)   // float3 纹理
float downsample1(int2 pixel, Tensor<float, 2> source)     // 单通道纹理
```

## 5. Processing Flow / 执行流程

```
1. 加载纹理 (同 Phase 1.2)

2. 编译 step_2_1_gpu_downsample.slang (包含 render + downsample)

3. 每帧循环
   ├── module.render(pixel=..., material=...)
   │   └── GPU: 渲染全分辨率 BRDF
   ├── module.downsample3(call_id(), source, _result=dest)
   │   └── GPU: 每个输出像素并行 2×2 box filter
   ├── (重复 steps=2 次)
   ├── app.blit(downsampled, tonemap=True)
   └── app.present()
```

## 6. Comparison / 对比

| Aspect | Phase 2.0 | Phase 2.1 | Change |
|--------|-----------|-----------|--------|
| 降采样位置 | CPU (NumPy) | GPU (Slang) | 性能 100-1000x |
| 数据搬运 | GPU→CPU→GPU | 零搬运 | 消除瓶颈 |
| Box filter 实现 | `reshape.mean(axis=(1,3))` | `.getv()` 读取 4 像素 | GPU 并行化 |
| shader 文件 | step_1_2 (只有 render) | step_2_1 (render+downsample) | 多功能 |
| 代码行数 (.slang) | 67 (只有 render) | 55 (render+downsample) | 更紧凑 |

## 7. Known Issues / 遗留问题

- 只支持 2×2 box filter，不够灵活 → Phase 2.2 引入完整 Mipmap 链
- downsample3 和 downsample1 是重复代码 (仅通道数不同) → 后续可以用 generic 优化
- 没做 LOD (Level of Detail) 选择 → Phase 2.2 多级分辨率输出
