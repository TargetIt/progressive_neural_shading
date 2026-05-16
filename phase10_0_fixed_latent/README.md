# Phase 10.0: Fixed Latent Texture

## Quick Start

```bash
python src/step_10_0_fixed_latent.py
```

使用固定 latent texture 作为网络输入。按 ESC 退出。

## What This Phase Teaches

- Latent texture: 比频率编码更紧凑的空间表示
- Bilinear sampling: GPU 硬件加速的纹理插值
- 为什么 latent 需要可训练: 固定的无法适配内容

## New in Phase 10.0

- **Latent Texture**: H×W×C 特征图
- **Bilinear sample**: 替代 sin/cos encoding

## Diff from Phase 9.1

| Phase 9.1 | Phase 10.0 |
|-----------|-----------|
| sin/cos encoding | Latent texture |
| Exploding dims | Compact dims |


## Using trace.py

```python
from trace import tensor_stats, print_stats, latent_stats

# 分析 latent texture: 分辨率和通道统计
stats = latent_stats(latent_tensor)
print(f"Latent: {stats['shape']}, channels={stats['channels']}")
```
