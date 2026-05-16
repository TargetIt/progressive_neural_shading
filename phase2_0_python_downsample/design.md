# Phase 1.0: Hello Slang — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — step_01 前半部分
> **前置 Phase**: 无 (第一个 Phase)

## 1. Introduction / 架构概览

Phase 1.0 是最小的可运行着色器：每个像素返回纯红色。
目标是理解 Slang 着色语言的基本结构和 slangpy 的 GPU 调用模型。

```
┌──────────────────────────────────────────────────┐
│                   step_1_0_hello.py               │
│  ┌──────────┐   ┌────────────┐   ┌───────────┐  │
│  │ App 框架 │ → │ Slang 编译 │ → │ GPU 执行  │  │
│  │ app.py   │   │ .slang→GPU │   │ 每像素并行 │  │
│  └──────────┘   └────────────┘   └───────────┘  │
│       │               │                │         │
│       v               v                v         │
│  窗口+设备       step.slang      512×512 红色    │
│  spy.Window      spy.Module       app.blit()     │
└──────────────────────────────────────────────────┘
```

## 2. Motivation / 设计动机

学习任何 GPU 着色语言的第一步是理解:
- **Shader 是什么**: 在 GPU 上对每个像素并行执行的函数
- **Host 端如何调用**: Python → slangpy → GPU driver → 编译 → 执行
- **数据流**: Tensor 分配 → shader 写入 → blit 到屏幕

如果直接在 step_01 中引入 BRDF + 纹理 + 法线贴图, 初学者会迷失在细节中。
Phase 1.0 剥离所有渲染知识, 只保留 GPU 编程的基本骨架。

## 3. Algorithm and Theory / 核心算法

### GPU 并行执行模型

```
CPU (Python):                     GPU (Slang):
  module.render(                   ┌─────────────────────┐
    pixel=spy.call_id(),           │ 对每个像素 (x,y):    │
    _result=output                 │   render(x,y)        │
  )                                │   并行执行 512×512 次 │
                                   └─────────────────────┘
```

### 关键概念

1. **spy.call_id()**: 告诉 slangpy 自动为 `pixel` 参数分配坐标
2. **Tensor**: GPU 上的多维数组, `spy.Tensor.empty()` 在 GPU 显存中分配
3. **blit**: 把 GPU Tensor 拷贝到屏幕输出纹理

## 4. Architecture / 架构

### 4.1 Module Breakdown

| 文件 | 职责 | 行数 |
|------|------|------|
| `app.py` | 窗口创建, GPU 设备, blit 到屏幕 | ~100 |
| `app.slang` | 最简 blit helper (Tensor→屏幕) | ~15 |
| `step_1_0_hello.slang` | 着色器: 返回纯红色 | ~20 |
| `step_1_0_hello.py` | 入口: 加载 shader, 渲染循环 | ~25 |

### 4.2 Key APIs

```python
# 创建 GPU 设备
device = spy.create_device(DeviceType.automatic, include_paths=[...])

# 编译 Shader
module = spy.Module.load_from_file(device, "shader.slang")

# 分配 GPU Tensor
tensor = spy.Tensor.empty(device, shape=(H, W), dtype=spy.float3)

# 逐像素执行 shader
module.render(pixel=spy.call_id(), _result=tensor)

# 显示到屏幕
app.blit(tensor)
```

## 5. Processing Flow / 执行流程

```
1. App.__init__()
   ├── spy.Window(512, 512)        ← 创建窗口
   ├── spy.create_device()         ← 创建 GPU 设备
   └── spy.Module.load(app.slang)  ← 加载 blit helper

2. step_1_0_hello.py
   ├── spy.Module.load(step.slang) ← 编译 shader (首次慢, ~2-5秒)
   └── spy.Tensor.empty(512,512)   ← 分配输出

3. while app.process_events():     ← 每帧循环
   ├── module.render(pixel=...)    ← GPU 执行: 512×512 次 render()
   ├── app.blit(output)            ← Tensor → 屏幕纹理
   └── app.present()               ← 提交帧
```

## 6. Comparison / 对比

| Aspect | 参考 step_01 | Phase 1.0 |
|--------|-------------|-----------|
| 着色器输出 | BRDF 光照结果 | 纯红色 |
| 纹理 | albedo + normal + roughness | 无 |
| MaterialParameters struct | ✅ | ❌ |
| BRDF 函数 | eval_brdf() | ❌ |
| 代码行数 (.slang) | 43 | 20 |
| 概念数 | 5+ | 2 |

## 7. Known Issues / 遗留问题

- 没有交互性 — 颜色是硬编码的, 不随输入变化
- 没有纹理 — 无法展示材质
- 下一 Phase (1.1) 将引入 BRDF 光照模型
