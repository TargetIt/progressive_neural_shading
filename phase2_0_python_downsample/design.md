# Phase 2.0: Python Downsample — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_02_mipmap (前半部分)
> **前置 Phase**: Phase 1.2

## 1. Introduction / 架构概览

Phase 2.0 在 Python/NumPy 端手动实现 2×2 box filter 降采样。
先用 Phase 1.2 的 Disney BRDF 渲染全分辨率图像，再在 CPU 端做降采样。

```
┌──────────────────────────────────────────────────┐
│                step_2_0_downsample.py              │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │ GPU 渲染 │ → │ CPU→NumPy  │ → │ Box Filter│  │
│  │ Phase 1.2│   │ to_numpy() │   │ 2x2 mean  │  │
│  └──────────┘   └────────────┘   └───────────┘  │
│       │               │                │         │
│       v               v                v         │
│  全分辨率 BRDF    numpy.ndarray   分辨率/4 输出   │
│  1024×512        reshape+mean    → spy.Tensor    │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

Phase 1.2 是全分辨率渲染，大窗口下性能不足。Mipmap 通过多级分辨率纹理链来解决:
- **为什么要降采样**: 远处的物体不需要全分辨率纹理
- **为什么先在 Python 做**: 让学习者理解 box filter 的数学原理，再迁移到 GPU
- **为什么要感受 CPU 性能**: `to_numpy()` 和 `from_numpy()` 有 GPU↔CPU 传输开销，这是理解 GPU 降采样价值的铺垫

## 3. Algorithm and Theory / 核心算法

### 3.1 2×2 Box Filter

```
Input:  H × W 图像
Output: H/2 × W/2 图像

对每个 2×2 块取平均:
output[i,j] = (input[2i,2j] + input[2i,2j+1] +
                input[2i+1,2j] + input[2i+1,2j+1]) / 4
```

### 3.2 NumPy 向量化实现

```python
reshaped = arr[:new_h*2, :new_w*2].reshape(new_h, 2, new_w, 2, -1)
result = reshaped.mean(axis=(1, 3))
```

### 3.3 Mipmap 链概念

```
Level 0: H × W        (原始分辨率)
Level 1: H/2 × W/2    (1 级降采样)
Level 2: H/4 × W/4    (2 级降采样)
...
```

## 4. Architecture / 架构

### 4.1 Module Breakdown

| 文件 | 职责 | 行数 |
|------|------|------|
| `app.py` | 窗口创建, GPU 设备, blit + tonemap | ~53 |
| `app.slang` | blit helper + ACES 色调映射 | ~30 |
| `brdf.slang` | 完整 Disney BRDF | ~118 |
| `step_1_2_full_brdf.slang` | 依赖: 全分辨率 BRDF 渲染 | ~67 |
| `step_2_0_downsample.py` | 入口: 渲染 + Python 降采样 | ~54 |
| `trace.py` | Tensor 统计 + 降采样验证 | ~45 |

### 4.2 关键数据流

```python
# GPU 渲染
output = Tensor.empty(device, shape)
module.render(..., _result=output)

# GPU → CPU → 降采样 → GPU
arr = output.to_numpy()                     # GPU → CPU (慢!)
downsampled = box_filter_2x2(arr)           # CPU 计算
result = Tensor.from_numpy(device, arr)     # CPU → GPU (慢!)
```

## 5. Processing Flow / 执行流程

```
1. 加载纹理 + 编译 shader (同 Phase 1.2)

2. 每帧循环
   ├── module.render(...)              ← GPU 渲染全分辨率 BRDF
   ├── output.to_numpy()               ← GPU → CPU 拷贝
   ├── reshape → 2x2 mean (steps=2)    ← CPU 降采样 (分辨率/4)
   ├── Tensor.from_numpy(downsampled)  ← CPU → GPU 拷贝
   ├── app.blit(downsampled, tonemap=True)
   └── app.present()
```

## 6. Comparison / 对比

| Aspect | Phase 1.2 | Phase 2.0 | Change |
|--------|-----------|-----------|--------|
| 渲染 | 全分辨率 BRDF | 全分辨率 BRDF (同) | — |
| 降采样 | 无 | Python box filter 2×2 | 新增 |
| 输出分辨率 | 原始 1024×1024 | 256×256 (steps=2) | 降低 |
| 计算位置 | GPU only | GPU + CPU | GPU→CPU→GPU |
| 性能瓶颈 | GPU shader | GPU↔CPU 传输 | 新增瓶颈 |

## 7. Known Issues / 遗留问题

- `to_numpy()` 和 `from_numpy()` 有显著的 GPU↔CPU 传输延迟 → Phase 2.1 将降采样移回 GPU
- Python box filter 是串行的 (虽然用了 NumPy 向量化) → Phase 2.1 GPU 并行降采样
- 降采样后输出显示在原始大小的窗口中，会拉伸 → Phase 2.2 用 Mipmap 链正确处理 LOD
