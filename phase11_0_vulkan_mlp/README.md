# Phase 11.0: Vulkan MLP Training

## Prerequisites

- Vulkan SDK 1.3+
- CMake 3.20+
- Ninja (or Make)
- C++17 compiler

## Build & Run

```bash
# Configure and build
cmake -S src -B _build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build _build

# Run training
_build/mlp-training
```

## What This Phase Teaches

- C++ host code with Slang API: 直接管理 GPU 设备和 shader
- Headless compute: 无图形窗口的 GPU 计算
- Production deployment: Python → C++ 迁移

## New in Phase 11.0

- **C++/Vulkan**: 替代 Python/slangpy
- **CMake 构建**: 替代 pip install
- **Headless training**: 纯计算，无渲染窗口

## Diff from Phase 10.2

| Phase 10.2 | Phase 11.0 |
|-----------|-----------|
| Python + slangpy | C++ + Slang API |
| pip install | CMake build |
| spy.Window | Headless |
| Interactive | Batch |

## Using trace.py

```python
from trace import verify_project

# 验证 C++/Vulkan 项目结构是否完整
ok, missing = verify_project('.')
if ok:
    print('Project structure OK')
else:
    print(f'Missing files: {missing}')
```
