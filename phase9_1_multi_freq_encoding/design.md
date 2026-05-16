# Phase 9.1: Multi-Frequency Encoding — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — network step_04 (完整版)
> **前置 Phase**: Phase 9.0

## 1. Introduction

Phase 9.1 是 Step 9 (Frequency Encoding) 最终形态，等价于 `network/step_04_frequency_encoding`。
NeRF-style 多频率位置编码: sin/cos 在多个频率尺度上的映射。

```
γ(v) = (sin(2^0 π v), cos(2^0 π v), sin(2^1 π v), cos(2^1 π v), ...)
For N frequencies: 2D input → 4N D output (sin+cos per dimension per freq)
```

## 2. Core Algorithm

```
freq = 2^k * π for k = 0..N-1
encoded_uv = [sin(freq*uv.x), cos(freq*uv.x),
              sin(freq*uv.y), cos(freq*uv.y)] for all freqs
```

### Effect
- 低频 (k=0): 捕捉全局光照变化
- 高频 (k=N-1): 捕捉精细纹理细节
- 多频率让网络自己选择关注哪个尺度

## 3. Comparison

| Aspect | Phase 9.0 | Phase 9.1 |
|--------|-----------|-----------|
| Frequencies | 1 | N (multi-scale) |
| Functions | sin only | sin + cos |
| 输入维度 | 2→2 | 2→4N |
| 参考对齐 | 中间 | 等价 step_04 |

## 4. Known Issues

- 编码维度爆炸 (N=10 → 40D input) → Step 10 用 latent texture 压缩
