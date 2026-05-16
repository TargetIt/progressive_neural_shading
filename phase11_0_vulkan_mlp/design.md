# Phase 11.0: Vulkan MLP — Design Document

> **对应**: SIGGRAPH 2025 Neural Shading Course — hardware-acceleration/mlp-training
> **前置 Phase**: Phase 10.2

## 1. Introduction

Phase 11.0 是硬件加速的第一步: 将神经网络训练从 Python/Slang 迁移到 C++/Vulkan 原生环境。
使用 Slang C++ API 直接创建 Vulkan 设备、编译 shader 和执行训练。

```
C++ Application
    ↓ slang API
Vulkan Device (GPU)
    ↓ slang compute shaders
MLP Training (forward + backward + Adam)
```

## 2. Key Differences from Phase 10

| Aspect | Phase 10.x | Phase 11.0 |
|--------|-----------|-----------|
| Host language | Python | C++ |
| GPU API | slangpy (Python binding) | Slang C++ API |
| Build system | pip install | CMake + Ninja |
| Window | spy.Window | Headless (compute only) |
| Execution | Interactive | Batch training |

## 3. Architecture

```
phase11_0_vulkan_mlp/
├── build.sh              # CMake configure + build + run
├── src/
│   ├── CMakeLists.txt    # CMake build configuration
│   └── mlp-training/
│       ├── mlp-training.cpp  # C++ host code
│       ├── network.slang     # Network struct definition
│       ├── kernels.slang     # Compute shader kernels
│       ├── adam.slang        # Adam optimizer
│       ├── common.slang      # Shared utilities
│       ├── mlp_sw.slang      # Software MLP implementation
│       └── mlvec_sw.slang    # Software matrix-vector
└── tests/
    └── test_phase11_0.py     # Structure verification tests
```

## 4. Key Components

- **mlp-training.cpp**: Creates Vulkan device, loads slang modules, runs training loop
- **kernels.slang**: Compute shader entry points (forward, backward, optimizer_step)
- **network.slang**: Network structure with parameter classes
- **adam.slang**: Adam optimizer implementation in Slang

## 5. Known Issues

- 仅用软件模拟的 MLP (SW = software) → Phase 11.1 使用 CoopVec 硬件加速
- 需要 Vulkan SDK 和 CMake 3.20+
