# Phase 11.0 Requirements

## New Features (2026-05-16)

- **C++/Vulkan 原生训练**: Slang C++ API 直接驱动 GPU
- **CMake 构建系统**: 跨平台编译配置
- **Headless 训练**: 无窗口，纯计算模式

## Functional Requirements

1. CMake configure + build 成功 (需要 Vulkan SDK)
2. C++ host 代码正确调用 Slang API
3. 训练循环可执行 (forward + backward + Adam)

## Acceptance Criteria

- [x] CMakeLists.txt configures correctly
- [x] All .slang files compile without errors
- [x] Test suite verifies project structure
