# Phase 9.0: Sin Encoding

## Quick Start

```bash
python src/step_9_0_sin_encoding.py
```

Sin 位置编码: UV → sin(freq * UV) → MLP。按 ESC 退出。

## What This Phase Teaches

- Spectral bias: MLP 天然偏好低频函数
- 位置编码: 将低维坐标映射到高频空间
- NeRF-style encoding 的第一步

## New in Phase 9.0

- **sin(freq * UV)**: 单频率位置编码
- **高频细节改善**: 比 raw UV 更好的纹理重建

## Diff from Phase 8.1

| Phase 8.1 | Phase 9.0 |
|-----------|-----------|
| Raw UV input | sin(freq * UV) |
| Activation focus | Encoding focus |


## Using trace.py

```python
from trace import tensor_stats, print_stats, encoding_stats

# 分析 sin 编码输出: 范围、幅值
stats = encoding_stats(encoded_output)
print(f"Range: {stats['range']}, Mean abs: {stats['mean_abs']:.3f}")
```
