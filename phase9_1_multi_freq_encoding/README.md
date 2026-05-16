# Phase 9.1: Multi-Frequency Encoding

## Quick Start

```bash
python src/step_9_1_multi_freq_encoding.py
```

NeRF-style 多频率 sin/cos 编码。按 ESC 退出。

## What This Phase Teaches

- NeRF 位置编码: γ(v) = (sin, cos) at multiple frequency scales
- Multi-scale representation: 低频=全局, 高频=细节
- Why sin+cos: 提供相位信息，比单独 sin 更丰富

## New in Phase 9.1

- **Multi-frequency sin/cos**: N frequencies → 4N dim output
- **Scale pyramid**: 2^0π to 2^(N-1)π

## Diff from Phase 9.0

| Phase 9.0 | Phase 9.1 |
|-----------|-----------|
| Single freq, sin only | Multi-freq, sin+cos |
| 2D→2D | 2D→4N D |
