# Phase 11.1: CoopVec Acceleration — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — hardware-acceleration/mlp-training-coopvec
> **前置 Phase**: Phase 11.0

## 1. Introduction

Phase 11.1 是完整教程的最后一个 Phase。使用 NVIDIA Cooperative Vector (CoopVec) 硬件指令加速神经网络训练。比较软件 MLP (Phase 11.0) 和 CoopVec 加速版本 (Phase 11.1) 的性能差异。

```
Phase 11.0: SW matmul (mlp_sw.slang, mlvec_sw.slang)
Phase 11.1: HW matmul (Cooperative Vector intrinsics)
              ↓
         显著的训练加速 (2-5x)
```

## 2. Cooperative Vector Intrinsics

CoopVec 是 NVIDIA GPU 的硬件加速指令，专门针对神经网络推理/训练的矩阵运算:
- **硬件加速的 matmul**: 利用 Tensor Cores
- **Slang 内建支持**: `CoopVec` 类型和 `coopVecMatMul` 等 intrinsic
- **与 AD 兼容**: `[Differentiable]` 标注仍可使用

## 3. Architecture

```
phase11_1_coopvec/
├── build.sh
├── src/
│   ├── CMakeLists.txt
│   └── mlp-training-coopvec/
│       ├── mlp-training-coopvec.cpp  # C++ host with CoopVec
│       ├── mlp.slang                 # CoopVec MLP implementation
│       ├── mlvec.slang               # CoopVec matrix-vector
│       ├── network.slang             # Network with CoopVec params
│       ├── kernels.slang             # CoopVec compute kernels
│       ├── adam.slang                # Adam optimizer
│       └── common.slang              # Shared utilities
└── tests/
```

## 4. Comparison

| Aspect | Phase 11.0 | Phase 11.1 |
|--------|-----------|-----------|
| Matmul | Software (SW) | CoopVec (HW) |
| GPU | Any Vulkan GPU | NVIDIA GPU only |
| Speed | Baseline | 2-5x faster |
| Files | mlp_sw, mlvec_sw | mlp, mlvec |

## 5. Platform Requirements

- Windows/Linux with NVIDIA GPU
- Vulkan SDK with CoopVec extension
- macOS not supported
