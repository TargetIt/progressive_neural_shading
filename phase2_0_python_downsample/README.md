# Phase 2.0: Python Manual Downsampling

## Quick Start

```bash
python src/step_2_0_downsample.py
```

应该看到降采样后的 BRDF 渲染 (1024×512 窗口，输出分辨率 /4)。按 ESC 退出。

## What This Phase Teaches

- Mipmap 概念: 多级分辨率纹理链
- 2×2 box filter: 最简单的降采样算法 (4 个相邻像素取平均)
- GPU↔CPU 数据传输开销: `to_numpy()` / `from_numpy()` 的实际延迟
- 为什么降采样需要放在 GPU: 避免 PCIe 传输瓶颈

## New in Phase 2.0

- **downsample_python()**: NumPy reshape + mean 实现 2×2 box filter
- **多级降采样**: steps 参数支持多次降采样
- **GPU↔CPU 传输**: Tensor ↔ numpy.ndarray

## Diff from Phase 1.2

| Phase 1.2 | Phase 2.0 |
|-----------|-----------|
| 全分辨率输出 | 降采样后输出 (分辨率/4) |
| GPU only | GPU 渲染 + CPU 降采样 |
| 无数据搬运 | GPU↔CPU 往返传输 |
| 1024×1024 窗口 | 1024×512 窗口 |
