# Phase 2.2: Mipmap Chain — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_02_mipmap (完整版)
> **前置 Phase**: Phase 2.1

## 1. Introduction / 架构概览

Phase 2.2 是 Step 2 的最终形态，功能等价于参考项目 `neural-shading-s25/mipmap/step_02_mipmap`。
建立完整的 GPU Mipmap 降采样链: 全分辨率渲染 → 多级 2×2 box filter → 输出。

```
┌──────────────────────────────────────────────────┐
│                  step_2_2_mipmap.py                 │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │ BRDF渲染 │ → │ Mipmap链   │ → │ 显示      │  │
│  │ render() │   │downsample()│   │ blit()    │  │
│  └──────────┘   └────────────┘   └───────────┘  │
│       │               │                │         │
│       v               v                v         │
│  全分辨率 BRDF   Level 0: H×W      降采样输出    │
│  1024×1024      Level 1: H/2×W/2   Level 2      │
│                 Level 2: H/4×W/4   (显示)      │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

Phase 2.1 实现了 GPU 降采样，但缺少对 mipmap 作为"链"的概念理解。
- **什么是 Mipmap 链**: 连续的多级分辨率，每级是上一级的 1/4 像素数
- **为什么需要链**: 不同 shader 调用可能需要不同 LOD (Level of Detail)
- **与参考代码的对齐**: 功能等价于 `step_02_mipmap.py`，完整的 Step 2 交付

## 3. Algorithm and Theory / 核心算法

### 3.1 Mipmap 分辨率链

```
Level 0: 原始 (1024×1024)
Level 1: 1/2  (512×512)
Level 2: 1/4  (256×256)
Level 3: 1/8  (128×128)
...
```

### 3.2 迭代式降采样

```python
def downsample(source, steps):
    for _ in range(steps):
        dest = Tensor.empty(shape=(H//2, W//2))
        if dtype == vector:
            module.downsample3(call_id(), source, _result=dest)
        else:
            module.downsample1(call_id(), source, _result=dest)
        source = dest
    return source
```

## 4. Architecture / 架构

### 4.1 Module Breakdown

| 文件 | 职责 | 行数 |
|------|------|------|
| `app.py` | 窗口创建, GPU 设备, blit | ~53 |
| `app.slang` | blit helper + ACES 色调映射 | ~30 |
| `brdf.slang` | 完整 Disney BRDF | ~118 |
| `step_2_2_mipmap.slang` | MaterialParameters + render() + downsample3/1 | ~46 |
| `step_2_2_mipmap.py` | 入口: 渲染 + mipmap 链降采样 | ~43 |
| `trace.py` | Tensor 统计 + mipmap 链验证 | ~52 |

## 5. Processing Flow / 执行流程

```
1. 加载 3 张 PBR 纹理到 GPU
2. 编译 step_2_2_mipmap.slang

3. 每帧循环
   ├── allocate output (H×W)
   ├── module.render(pixel, material, light_dir, view_dir)
   │   └── GPU: Disney BRDF 全分辨率渲染
   ├── downsample(output, steps=2)
   │   ├── iteration 1: module.downsample3() → H/2 × W/2
   │   └── iteration 2: module.downsample3() → H/4 × W/4
   ├── app.blit(result)
   └── app.present()
```

## 6. Comparison / 对比

| Aspect | Phase 2.1 | Phase 2.2 | Change |
|--------|-----------|-----------|--------|
| 降采样方法 | GPU downsample | GPU downsample (同) | — |
| Mipmap 概念 | 2 级降采样 | 完整的 mipmap 链 | 概念完整性 |
| 参考对齐 | 中间状态 | 等价 step_02_mipmap | 达到里程碑 |
| 窗口 | 1024×512 | 1024×1024 | 全分辨率 |
| blit | 有 tonemap | 无 tonemap (clamp) | 简化 |

## 7. Known Issues / 遗留问题

- 对纹理本身做 mipmap，而非仅对输出 → 参考代码的完整做法
- 没有真正的 LOD 选择机制 (选择哪一级 mipmap)
- 下一 Step 3 将引入超采样 (SSAA) 以改善抗锯齿
