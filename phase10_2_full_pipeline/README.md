# Phase 10.2: Full Pipeline

## Quick Start

```bash
python src/step_10_2_full_pipeline.py
```

完整神经纹理管线: Latent → MLP → Material → BRDF → Render。按 ESC 退出。

## What This Phase Teaches

- 完整神经纹理: 从 latent texture 到 BRDF 输出的端到端管线
- 可微渲染: 所有组件都是可微的
- Neural Shading 的核心: MLP 学习 PBR 材质参数

## New in Phase 10.2

- **Latent → Material → BRDF 管线**
- **End-to-end differentiable rendering**

## Diff from Phase 10.1

| Phase 10.1 | Phase 10.2 |
|-----------|-----------|
| Latent → RGB | Latent → Material → BRDF → RGB |
| Direct color | PBR pipeline |


## Using trace.py

```python
from trace import tensor_stats, print_stats, pipeline_stats

# 报告完整管线进度 (loss, step, lr, latent)
stats = pipeline_stats(loss, step, lr, latent, network)
print(f"Step {stats['step']} | Loss: {stats['loss']:.6f}")
```
