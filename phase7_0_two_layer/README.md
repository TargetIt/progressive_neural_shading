# Phase 7.0: Two-Layer Forward

## Quick Start

```bash
python src/step_7_0_two_layer.py
```

双层网络: 2→16→3 forward pass。按 ESC 退出。

## What This Phase Teaches

- 多层网络 (MLP): 堆叠多个 NetworkParameters
- 隐藏层概念: 16 个隐藏神经元增加模型容量
- 层间 ReLU: 只有隐藏层有 activation，输出层是线性的

## New in Phase 7.0

- **layer0 + layer1**: 第一个多层网络
- **隐藏层**: 16 维中间表示

## Diff from Phase 6.2

| Phase 6.2 | Phase 7.0 |
|-----------|-----------|
| Single layer | Two layers |
| 2→3 directly | 2→16→3 |
| Training | Forward only |


## Using trace.py

```python
from trace import tensor_stats, print_stats, layer_stats

# 逐层打印网络参数
layer_stats(network)
# => Layer 0: W(16x2) |W|=0.123
# => Layer 1: W(3x16) |W|=0.089
```
