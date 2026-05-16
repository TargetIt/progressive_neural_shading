# Phase 8.1: Activation Gallery

## Quick Start

```bash
python src/step_8_1_activation_gallery.py
```

并排对比多种激活函数。按 ESC 退出。

## What This Phase Teaches

- 激活函数对比: ReLU vs LeakyReLU vs GELU vs Swish
- Dead neuron 问题及解决方案
- 现代激活函数的设计思路

## New in Phase 8.1

- **Activation Gallery**: 5+ 种激活函数可视化
- **LeakyReLU/GELU/Swish**: 解决 ReLU dead neuron

## Diff from Phase 8.0

| Phase 8.0 | Phase 8.1 |
|-----------|-----------|
| ReLU analysis | Multi-activation gallery |
| Single | Comparative |


## Using trace.py

```python
from trace import tensor_stats, print_stats, activation_gallery_stats

# 对比多种激活函数的输出
activation_gallery_stats({'ReLU': out_relu, 'LeakyReLU': out_leaky, 'GELU': out_gelu})
```
