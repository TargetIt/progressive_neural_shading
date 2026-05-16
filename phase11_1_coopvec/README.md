# Phase 11.1: CoopVec Hardware Acceleration

## Prerequisites

- NVIDIA GPU (Cooperative Vector compatible)
- Vulkan SDK 1.3+ with CoopVec extension
- CMake 3.20+, Ninja, C++17 compiler

## Build & Run

```bash
# Configure and build
cmake -S src -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _build

# Run CoopVec-accelerated training
_build/mlp-training-coopvec
```

## What This Phase Teaches

- Cooperative Vector: NVIDIA GPU 硬件加速神经网络
- HW vs SW matmul: 性能差异和精度对比
- Production deployment: 从 Python → C++ → HW-accelerated 的完整路径

## New in Phase 11.1

- **CoopVec matmul**: 硬件加速矩阵乘法
- **NVIDIA Tensor Core**: 通过 CoopVec 访问
- **2-5x speedup**: vs Phase 11.0 SW matmul

## Diff from Phase 11.0

| Phase 11.0 | Phase 11.1 |
|-----------|-----------|
| Software matmul | CoopVec HW matmul |
| Any Vulkan GPU | NVIDIA only |
| SW speed | 2-5x faster |

## Using trace.py

```python
from trace import verify_project

# 验证 CoopVec 项目结构是否完整
ok, missing = verify_project('.')
if ok:
    print('Project structure OK')
else:
    print(f'Missing files: {missing}')
```
