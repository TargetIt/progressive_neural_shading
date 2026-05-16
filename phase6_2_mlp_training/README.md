# Phase 1.0: Hello Slang — Minimal Shader

## Quick Start

```bash
pip install slangpy
python src/step_1_0_hello.py
```

应该看到 512×512 的纯红色窗口。按 ESC 退出。

## What This Phase Teaches

- Slang `.slang` 文件的基本结构 (`import slangpy;`, 函数定义)
- `slangpy` 的 GPU 调用模型 (`spy.call_id()`, `Tensor`, `blit`)
- GPU 并行执行: `render(pixel)` 对每个像素执行一次, 512×512 = 262,144 次并行

## New in Phase 1.0

- **app.py**: 最简渲染框架 (窗口 + GPU 设备 + blit)
- **app.slang**: 最简 blit helper
- **step_1_0_hello.slang**: 返回纯红色的着色器
- **trace.py**: Tensor 统计 (min/max/mean)
