# Phase 3.0: Higher Resolution Render

## Quick Start

```bash
python src/step_3_0_hi_res.py
```

看到左右并排对比: 左半 1x BRDF，右半 2x 超采样+降采样 (更平滑)。按 ESC 退出。

## What This Phase Teaches

- 超采样原理: 高分辨率渲染 + 降采样 = 抗锯齿
- `blit()` 的 offset/size 参数: 在窗口中定位多个输出
- 画质对比: 1x vs 2x 的视觉差异

## New in Phase 3.0

- **2x 超采样渲染**: `Tensor.empty(2H, 2W)` 的高分辨率输出
- **并排 blit**: 两个 `blit()` 调用分别输出到左右半屏

## Diff from Phase 2.2

| Phase 2.2 | Phase 3.0 |
|-----------|-----------|
| 单次渲染 | 两次渲染 (1x + 2x) |
| 全屏 blit | 左右并排 blit |
| Mipmap 链 | 超采样预览 |


## Using trace.py

```python
from trace import tensor_stats, print_stats, compare_1x_vs_2x

# 对比 1x 和 2x 超采样结果
ok, msg, diff = compare_1x_vs_2x(out_1x, out_2x_down)
print(f'1x vs 2x diff: {diff:.4f}')
```
