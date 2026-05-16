# Phase 2.1: GPU Downsampling

## Quick Start

```bash
python src/step_2_1_gpu_downsample.py
```

应该看到 GPU 降采样后的 BRDF 渲染 (相比 Phase 2.0 的 CPU 版本，运行更流畅)。按 ESC 退出。

## What This Phase Teaches

- GPU 并行降采样: 每个输出像素独立取出 4 个源像素取平均
- 为什么 GPU 降采样比 CPU 快: 避免 GPU↔CPU 数据传输
- Slang 中多函数 module: render + downsample 共存于一个 .slang 文件
- Tensor 维度处理: float3 (3 通道) vs float (单通道) 的分离函数

## New in Phase 2.1

- **downsample3()**: GPU 并行 2×2 box filter for float3
- **downsample1()**: GPU 并行 2×2 box filter for float
- **自包含 shader**: 渲染 + 降采样都在 step_2_1_gpu_downsample.slang

## Diff from Phase 2.0

| Phase 2.0 | Phase 2.1 |
|-----------|-----------|
| CPU (NumPy) 降采样 | GPU (Slang) 降采样 |
| GPU↔CPU 数据传输 | 零数据传输 |
| 慢 (PCIe 瓶颈) | 快 (GPU 并行) |
| 依赖 step_1_2 shader | 自包含 shader |


## Using trace.py

```python
from trace import tensor_stats, print_stats, verify_gpu_downsample

# 验证 GPU 降采样: 分辨率正确 + 无 NaN
ok, msg = verify_gpu_downsample(original, downsampled, steps=2)
print(msg)  # => shape=(512,512) OK
```
