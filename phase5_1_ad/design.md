# Phase 5.1: Automatic Differentiation — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — autodiff
> **前置 Phase**: Phase 5.0

## 1. Introduction

Phase 5.1 引入 Slang 的自动微分 (AD) 系统，替换 Phase 5.0 的数值梯度。
AD 通过编译器生成的 backward pass 计算精确梯度，速度远快于有限差分。

```
Phase 5.0 (数值): 渲染(rough+h) + 渲染(rough-h) → (L+ - L-)/(2h)
Phase 5.1 (AD):   bwd_diff(loss) → 一次 backward pass 得到所有梯度
```

## 2. Core Algorithm

### Slang AD 机制

```slang
struct MaterialParameters {
    RWTensor<float3, 2> albedo;
    RWTensor<float3, 2> albedo_grad;  // ← 梯度 Tensor

    [Differentiable]
    float3 get_albedo(int2 p) { return albedo.getv(p); }

    [BackwardDerivativeOf(get_albedo)]
    void get_albedo_bwd(int2 p, float3 grad) {
        albedo_grad.setv(p, grad);  // 累积梯度
    }
};

// 一次调用计算所有梯度
bwd_diff(loss)(pixel, reference, mat, light_dir, view_dir, 1);
```

### 关键标注
- `[Differentiable]`: 标记可微分函数
- `[BackwardDerivativeOf(f)]`: 定义 f 的反向传播规则
- `bwd_diff(f)(args, seed)`: 触发反向传播，seed=1 是 loss 对自身的导数

## 3. Architecture

| 文件 | 职责 |
|------|------|
| `step_5_1_ad.slang` | MaterialParameters (with AD) + render + loss + calculate_grad |
| `step_5_1_ad.py` | Trainable params init + bwd_diff call + gradient stats |
| `trace.py` | ad_gradient_stats() |

## 4. Comparison

| Aspect | Phase 5.0 | Phase 5.1 |
|--------|-----------|-----------|
| 梯度计算 | Central finite diff | bwd_diff (AD) |
| 每参数 cost | 2 次前向 pass | 1 次 backward pass |
| 精度 | 近似 (O(h^2)) | 精确 (machine precision) |
| RWTensor | No | Yes (可读写 Tensor) |
| BackwardDerivativeOf | No | Yes |

## 5. Known Issues

- 需要初始化梯度 Tensor (`Tensor.zeros_like`)，且每帧 `zero_()`
- `no_diff` 标注避免对 light_dir/view_dir 计算不必要的梯度
- 下一 Phase 5.2 将使用 AD 梯度进行 Adam 优化
