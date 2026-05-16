# Phase 7.1: Deep Network Training

## Quick Start

```bash
python src/step_7_1_deep_network.py
```

3 层深度网络 (32→32) 训练。按 ESC 退出。

## What This Phase Teaches

- 深度网络: more layers = more expressiveness
- 隐藏层维度: 32 vs 16 (Phase 7.0) 的效果差异
- 多层反向传播: 梯度通过多个层回传

## New in Phase 7.1

- **3 层 MLP**: 2→32→32→3
- **完整多层训练管线**

## Diff from Phase 7.0

| Phase 7.0 | Phase 7.1 |
|-----------|-----------|
| 2 layers (2→16→3) | 3 layers (2→32→32→3) |
| Forward only | Full training |


## Using trace.py

```python
from trace import tensor_stats, print_stats, deep_network_stats

# 报告深层网络 (3层) 参数
deep_network_stats(network)
# => L0(2->32): |W|=0.234  L1(32->32): |W|=0.156
```
