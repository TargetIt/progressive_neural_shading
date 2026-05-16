# Phase 2.2: Mipmap Chain

## Quick Start

```bash
python src/step_2_2_mipmap.py
```

应该看到 1024×1024 的降采样 BRDF 渲染。按 ESC 退出。

## What This Phase Teaches

- Mipmap 链的概念: 多级分辨率纹理，每级 1/4 像素数
- 迭代 GPU 降采样: for 循环驱动多次 downsample() 调用
- 为什么需要 mipmap: 远处物体用低分辨率，节省带宽和计算
- Step 2 里程碑: 功能等价于参考 `step_02_mipmap`

## New in Phase 2.2

- **完整 mipmap 降采样链**: 可调 steps 的多级降采样
- **全分辨率窗口**: 1024×1024

## Diff from Phase 2.1

| Phase 2.1 | Phase 2.2 |
|-----------|-----------|
| GPU 降采样 | GPU 降采样链 |
| 1024×512 | 1024×1024 |
| blit tonemap | blit clamp |
| 中间状态 | 等价参考 step_02 |
