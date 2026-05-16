# Phase 3.1: SSAA Pipeline

## Quick Start

```bash
python src/step_3_1_ssaa.py
```

左半: 标准 1x BRDF。右半: 4×4 SSAA (16 子像素平均)。按 ESC 退出。

## What This Phase Teaches

- SSAA (Super-Sampling Anti-Aliasing): 高分辨率渲染后降采样
- 4×4 sub-pixel sampling: 每像素 16 个采样点
- SSAA 作为 Ground Truth 参考: 后续 Phase 的神经网络目标

## New in Phase 3.1

- **ssaa_render()**: 4×4 子像素循环采样
- **完整 SSAA 管线**: 等价于参考 step_03_supersample

## Diff from Phase 3.0

| Phase 3.0 | Phase 3.1 |
|-----------|-----------|
| 2x 超采样 | 4×4 = 16x 子像素 |
| 单 render | ssaa_render + render |
| 概念演示 | 完整 SSAA 管线 |


## Using trace.py

```python
from trace import tensor_stats, print_stats, compare_ssaa

# 对比标准渲染和 SSAA 结果
ok, msg, diff = compare_ssaa(std_1x, ssaa_result)
print(msg)
```
